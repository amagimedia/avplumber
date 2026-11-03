const test = require('node:test');
const assert = require('node:assert/strict');
const http = require('http');
const { publicAddress } = require('../backend/publicAddress.js');

// A metadata server answering `routes` ({"METHOD /path": [status, body]}), and its endpoints.
async function metadata(routes) {
  const seen = [];
  const server = http.createServer((req, res) => {
    seen.push({ route: `${req.method} ${req.url}`, headers: req.headers });
    const [status, body] = routes[`${req.method} ${req.url}`] || [404, ''];
    res.writeHead(status).end(body);
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  return { server, seen, endpoints: { gce: `${base}/gce`, awsToken: `${base}/token`, aws: `${base}/aws` } };
}

test('GCE metadata gives the address, asked with its flavor header', async () => {
  const m = await metadata({ 'GET /gce': [200, '203.0.113.10\n'] });
  assert.equal(await publicAddress(m.endpoints), '203.0.113.10');
  assert.equal(m.seen[0].headers['metadata-flavor'], 'Google');
  m.server.close();
});

test('without GCE, AWS IMDSv2: a token first, then the address with it', async () => {
  const m = await metadata({ 'PUT /token': [200, 'tok'], 'GET /aws': [200, '198.51.100.20'] });
  assert.equal(await publicAddress(m.endpoints), '198.51.100.20');
  assert.equal(m.seen.find((s) => s.route === 'GET /aws').headers['x-aws-ec2-metadata-token'], 'tok');
  m.server.close();
});

test('no address off a cloud, without a public IP, or from a malformed answer', async () => {
  for (const routes of [{}, { 'PUT /token': [200, 'tok'] }, { 'GET /gce': [200, '<html>'] }]) {
    const m = await metadata(routes);
    assert.equal(await publicAddress(m.endpoints), null, JSON.stringify(routes));
    m.server.close();
  }
  assert.equal(await publicAddress({ gce: 'http://127.0.0.1:9/x', awsToken: 'http://127.0.0.1:9/t', aws: 'http://127.0.0.1:9/a' }), null);
});
