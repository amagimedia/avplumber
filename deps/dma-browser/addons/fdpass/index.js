'use strict';

const path = require('node:path');
const binary = require('node-gyp-build');

const addon = binary(path.join(__dirname));

async function sendFd(socketPath, fd) {
  return new Promise((resolve, reject) => {
    try {
      addon.sendFd(socketPath, fd);
      resolve();
    } catch (err) {
      reject(err);
    }
  });
}

async function sendFdWithInfo(socketPath, fd, texInfoBuffer) {
  return new Promise((resolve, reject) => {
    try {
      addon.sendFdWithInfo(socketPath, fd, texInfoBuffer);
      resolve();
    } catch (err) {
      reject(err);
    }
  });
}

function createServer(socketPath) {
  return addon.createServer(socketPath);
}

async function broadcastFd(socketPath, fd, texInfoBuffer) {
  return new Promise((resolve, reject) => {
    try {
      if (texInfoBuffer && typeof addon.broadcastFdWithInfo === 'function') {
        addon.broadcastFdWithInfo(socketPath, fd, texInfoBuffer);
      } else {
        addon.broadcastFd(socketPath, fd, texInfoBuffer);
      }
      resolve();
    } catch (err) {
      reject(err);
    }
  });
}

function closeServer(socketPath) {
  return addon.closeServer(socketPath);
}

function setServerLogger(socketPath, callback) {
  return addon.setServerLogger(socketPath, callback);
}

function close() {
  if (typeof addon.close === 'function') addon.close();
}

module.exports = {
  sendFd,
  sendFdWithInfo,
  createServer,
  broadcastFd,
  closeServer,
  setServerLogger,
  close,
};
