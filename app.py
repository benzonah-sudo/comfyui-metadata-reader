from flask import Flask, request, jsonify, render_template
import struct
import json
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024


def read_png_text_chunks(data):
    PNG_SIG = b'\x89PNG\r\n\x1a\n'
    if not data.startswith(PNG_SIG):
        return {}
    chunks = {}
    pos = 8
    while pos < len(data) - 12:
        if pos + 8 > len(data):
            break
        length = __import__('struct').unpack('>I', data[pos:pos+4])[0]
        chunk_type = data[pos+4:pos+8].decode('ascii', errors='replace')
        chunk_data = data[pos+8:pos+8+length]
        if chunk_type == 'tEXt':
            nul = chunk_data.find(b'\x00')
            if nul >= 0:
                chunks[chunk_data[:nul].decode('latin-1')] = chunk_data[nul+1:].decode('latin-1')
        elif chunk_type == 'iTXt':
            nul = chunk_data.find(b'\x00')
            if nul >= 0:
                key = chunk_data[:nul].decode('utf-8', errors='replace')
                rest = chunk_data[nul+1:]
                n2 = rest.find(b'\x00', 2)
                if n2 >= 0:
                    n3 = rest.find(b'\x00', n2+1)
                    if n3 >= 0:
                        chunks[key] = rest[n3+1:].decode('utf-8', errors='replace')
        if chunk_type == 'IEND':
            break
        pos += 12 + length
    return chunks


def extract_summary(prompt):
    rows = []
    if not prompt:
        return rows
    for node in prompt.values():
        cls = node.get('class_type', '')
        inp = node.get('inputs', {})
        if cls in ('KSampler', 'KSamplerAdvanced'):
            for k in ('seed','steps','cfg','sampler_name','scheduler','denoise'):
                if k in inp:
                    rows.append({'key': k.replace('_',' ').title(), 'value': str(inp[k])})
        if cls in ('CheckpointLoaderSimple','CheckpointLoader'):
            if 'ckpt_name' in inp:
                rows.append({'key': 'Checkpoint', 'value': inp['ckpt_name']})
        if cls == 'LoraLoader' and 'lora_name' in inp:
            rows.append({'key': 'LoRA', 'value': str(inp['lora_name'])})
        if cls == 'EmptyLatentImage' and 'width' in inp:
            rows.append({'key': 'Resolution', 'value': str(inp['width']) + ' x ' + str(inp['height'])})
        if cls == 'CLIPTextEncode' and isinstance(inp.get('text'), str):
            rows.append({'key': 'Text prompt', 'value': inp['text']})
    return rows


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/parse', methods=['POST'])
def parse_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    if not f.filename.lower().endswith('.png'):
        return jsonify({'error': 'Only PNG files are supported'}), 400
    data = f.read()
    chunks = read_png_text_chunks(data)
    has_prompt = 'prompt' in chunks
    has_workflow = 'workflow' in chunks
    if not has_prompt and not has_workflow:
        return jsonify({'error': 'No ComfyUI metadata found in this image'}), 200
    prompt_obj = None
    workflow_obj = None
    if has_prompt:
        try:
            prompt_obj = json.loads(chunks['prompt'])
        except:
            pass
    if has_workflow:
        try:
            workflow_obj = json.loads(chunks['workflow'])
        except:
            pass
    summary = extract_summary(prompt_obj)
    nodes = []
    if prompt_obj:
        for node_id, node in prompt_obj.items():
            inputs = {k: ('[node ' + str(v[0]) + ', slot ' + str(v[1]) + ']') if isinstance(v, list) else str(v) for k, v in (node.get('inputs') or {}).items()}
            nodes.append({'id': node_id, 'class_type': node.get('class_type','Unknown'), 'inputs': inputs})
    return jsonify({'has_prompt': has_prompt, 'has_workflow': has_workflow, 'summary': summary, 'nodes': nodes, 'prompt_json': json.dumps(prompt_obj, indent=2) if prompt_obj else None, 'workflow_json': json.dumps(workflow_obj, indent=2) if workflow_obj else None, 'filename': f.filename, 'size_kb': round(len(data)/1024, 1)})


if __name__ == '__main__':
    print("ComfyUI Metadata Reader running at http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)
