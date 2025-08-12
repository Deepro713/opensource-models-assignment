
ollama create hs-gemma -f Modelfile
ollama run hs-gemma

ollama list

ollama serve
curl http://localhost:11434/api/generate -d '{"model": "gemma3:1b", "prompt": "Hello, how can I assist you today?"}'

https://github.com/ollama/ollama-python
