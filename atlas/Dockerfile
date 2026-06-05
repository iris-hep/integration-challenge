FROM --platform=linux/amd64 ghcr.io/prefix-dev/pixi:latest

WORKDIR /app

COPY pixi.toml pixi.lock ./
RUN pixi install --frozen

COPY . .

EXPOSE 8888
ENV PIXI_PROJECT_MANIFEST=/app/pixi.toml

CMD ["pixi", "run", "jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]