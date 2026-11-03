'use strict';
// The public IPv4 address of the cloud VM this runs on, from its instance metadata: GCE, then AWS
// (IMDSv2). WEBUI_ALLOWED_HOSTS=auto allows it as a Host, since a cloud VM's ephemeral address
// changes when it stops and starts while a container keeps its environment. The Janus image
// resolves JANUS_NAT_1_1=auto the same way (docker-compose/images/janus/janus-entrypoint.sh).
const http = require('http');

const METADATA = {
  gce: 'http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip',
  awsToken: 'http://169.254.169.254/latest/api/token',
  aws: 'http://169.254.169.254/latest/meta-data/public-ipv4',
};
const IPV4 = /^\d{1,3}(\.\d{1,3}){3}$/;

// The body of a 200 response, or null on any other answer, an error or after `timeoutMs`.
function fetchText(url, { method = 'GET', headers = {}, timeoutMs = 2000 } = {}) {
  return new Promise((resolve) => {
    const req = http.request(url, { method, headers, timeout: timeoutMs }, (res) => {
      let body = '';
      res.setEncoding('utf8');
      res.on('data', (chunk) => { body += chunk; });
      res.on('end', () => resolve(res.statusCode === 200 ? body.trim() : null));
    });
    req.on('timeout', () => req.destroy());
    req.on('error', () => resolve(null));
    req.end();
  });
}

// The address, or null off a cloud (or on one without a public IPv4 address).
async function publicAddress(endpoints = METADATA) {
  const gce = await fetchText(endpoints.gce, { headers: { 'Metadata-Flavor': 'Google' } });
  if (gce && IPV4.test(gce)) return gce;
  const token = await fetchText(endpoints.awsToken,
    { method: 'PUT', headers: { 'X-aws-ec2-metadata-token-ttl-seconds': '60' } });
  if (!token) return null;
  const aws = await fetchText(endpoints.aws, { headers: { 'X-aws-ec2-metadata-token': token } });
  return aws && IPV4.test(aws) ? aws : null;
}

module.exports = { publicAddress };
