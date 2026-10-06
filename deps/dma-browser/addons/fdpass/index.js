'use strict';

const path = require('node:path');
const binary = require('node-gyp-build');

const addon = binary(path.join(__dirname));

function createServer(socketPath) {
  return addon.createServer(socketPath);
}

async function broadcastFd(socketPath, fd, texInfoBuffer) {
  return addon.broadcastFd(socketPath, fd, texInfoBuffer);
}

function closeServer(socketPath) {
  return addon.closeServer(socketPath);
}

function setServerLogger(socketPath, callback) {
  return addon.setServerLogger(socketPath, callback);
}

function setReleaseCallback(socketPath, callback) {
  return addon.setReleaseCallback(socketPath, callback);
}

function monotonicTimeNs() {
  return addon.monotonicTimeNs();
}

module.exports = {
  createServer,
  broadcastFd,
  closeServer,
  setServerLogger,
  setReleaseCallback,
  monotonicTimeNs,
};
