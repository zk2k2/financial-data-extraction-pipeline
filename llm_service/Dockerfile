# llm_service/Dockerfile
FROM ollama/ollama:latest       
# expose Ollama API
ENV OLLAMA_HOST=0.0.0.0:8080
ENV OLLAMA_MODELS=/models

EXPOSE 8080
CMD ["ollama", "serve", "--listen", "0.0.0.0:8080"]
