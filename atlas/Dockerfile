FROM --platform=linux/amd64 ghcr.io/prefix-dev/pixi:latest

ARG ENVIRONMENT="default"
WORKDIR /app
COPY pixi.toml pixi.lock ./
RUN pixi install --locked --environment $ENVIRONMENT
# Activate ENVIRONMENT at container runtime by using the Pixi environment
# activation script as the container ENTRYPOINT
RUN echo "#!/usr/bin/env bash" > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh && \
    pixi shell-hook --environment $ENVIRONMENT --shell bash >> /app/entrypoint.sh && \
    echo 'exec "$@"' >> /app/entrypoint.sh

COPY . .

EXPOSE 8888
ENV PIXI_PROJECT_MANIFEST=/app/pixi.toml

CMD ["pixi", "run", "jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
ENTRYPOINT [ "/app/entrypoint.sh" ]