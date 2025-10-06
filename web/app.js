async function getHealth() {
  try {
    const res = await fetch('/health');
    const ok = res.ok ? 'ok' : 'error';
    document.getElementById('health').textContent = ok;
  } catch (e) {
    document.getElementById('health').textContent = 'error';
  }
}

function setResult(el, kind, msg) {
  el.className = 'result ' + kind;
  el.textContent = msg;
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

document.addEventListener('DOMContentLoaded', () => {
  getHealth();

  const compressForm = document.getElementById('compress-form');
  const compressResult = document.getElementById('compress-result');
  compressForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    setResult(compressResult, 'info', 'Compressing…');

    const fd = new FormData();
    const threshold = document.getElementById('compress-threshold').value;
    const chunkSize = document.getElementById('compress-chunk-size').value;
    const text = document.getElementById('compress-text').value.trim();
    const file = document.getElementById('compress-file').files[0];

    fd.append('threshold', threshold);
    fd.append('chunk_size', chunkSize);
    if (file) {
      fd.append('file', file);
    } else if (text) {
      fd.append('text', text);
    } else {
      setResult(compressResult, 'error', 'Provide text or upload a file.');
      return;
    }

    try {
      const res = await fetch('/compress', { method: 'POST', body: fd });
      if (!res.ok) {
        const err = await res.text();
        setResult(compressResult, 'error', 'Error: ' + err);
        return;
      }
      const blob = await res.blob();
      const cd = res.headers.get('Content-Disposition');
      const filename = (cd && cd.split('filename=')[1]) ? cd.split('filename=')[1] : 'compressed.llmc';
      downloadBlob(blob, filename.replace(/"/g, ''));
      const orig = res.headers.get('X-Original-Size');
      const gz = res.headers.get('X-Compressed-Size');
      const ratio = res.headers.get('X-Compression-Ratio');
      const extra = (orig && gz && ratio) ? ` (compressed=${gz}B, original=${orig}B, ratio=${ratio})` : '';
      setResult(compressResult, 'success', 'Downloaded ' + filename + extra);
    } catch (err) {
      setResult(compressResult, 'error', 'Network error');
    }
  });

  const decompressForm = document.getElementById('decompress-form');
  const decompressResult = document.getElementById('decompress-result');
  decompressForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    setResult(decompressResult, 'info', 'Decompressing…');

    const fd = new FormData();
    const threshold = document.getElementById('decompress-threshold').value;
    const chunkSize = document.getElementById('decompress-chunk-size').value;
    const file = document.getElementById('decompress-file').files[0];
    const download = document.getElementById('decompress-download').checked;
    const filename = document.getElementById('decompress-filename').value.trim() || 'decompressed.txt';

    if (!file) {
      setResult(decompressResult, 'error', 'Upload a .npy.gz file.');
      return;
    }

    fd.append('file', file);
    fd.append('threshold', threshold);
    fd.append('chunk_size', chunkSize);
    if (download) {
      fd.append('download', 'true');
      fd.append('filename', filename);
    }

    try {
      const res = await fetch('/decompress', { method: 'POST', body: fd });
      if (!res.ok) {
        const err = await res.text();
        setResult(decompressResult, 'error', 'Error: ' + err);
        return;
      }
      if (download) {
        const blob = await res.blob();
        downloadBlob(blob, filename);
        setResult(decompressResult, 'success', 'Downloaded ' + filename);
      } else {
        const text = await res.text();
        setResult(decompressResult, 'success', text);
      }
    } catch (err) {
      setResult(decompressResult, 'error', 'Network error');
    }
  });
});


