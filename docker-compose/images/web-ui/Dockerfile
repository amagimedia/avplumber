FROM node:22-bookworm-slim

WORKDIR /opt/avplumber-web-ui

COPY tools/web-ui/package.json tools/web-ui/package-lock.json ./
RUN npm ci

COPY tools/web-ui ./
RUN npm run build \
    && npm prune --omit=dev

ENV NODE_ENV=production
ENV WEBUI_PORT=22222

CMD ["npm", "start"]
