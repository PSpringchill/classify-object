export default async function handler(req, res) {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }
  try {
    const { image_base64 } = req.body || {};
    if (!image_base64) {
      res.status(400).json({ error: 'image_base64 is required' });
      return;
    }

    const url = process.env.PREDICT_URL;
    if (!url) {
      res.status(500).json({ error: 'PREDICT_URL is not configured' });
      return;
    }

    const headers = { 'Content-Type': 'application/json' };
    const authHeader = process.env.API_AUTH_HEADER || 'Authorization';
    const token = process.env.API_TOKEN;
    if (token) headers[authHeader] = token.startsWith('Bearer ') ? token : `Bearer ${token}`;

    const upstream = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify({ image_base64 }),
    });

    const data = await upstream.json().catch(() => null);
    if (!upstream.ok) {
      res.status(upstream.status).json({ error: data?.error || 'Upstream error' });
      return;
    }
    res.status(200).json(data);
  } catch (err) {
    res.status(500).json({ error: err?.message || 'Internal error' });
  }
}
