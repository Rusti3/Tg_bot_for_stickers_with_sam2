FROM node:24-bookworm-slim

WORKDIR /app

COPY apps/control-plane/package.json ./
COPY apps/control-plane/tsconfig.json ./
COPY apps/control-plane/src ./src
COPY apps/control-plane/tests ./tests
COPY apps/control-plane/migrations ./migrations

RUN npm install \
    && npm run build

CMD ["npm", "run", "start:api"]
