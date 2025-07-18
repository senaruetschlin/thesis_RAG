How to test the generator 

1. To start the vllm server for Fin-R1 run: 
   vllm serve "./Fin-R1" --host 0.0.0.0 --port 8000 --served-model-name Fin-R1

2. Run test_generator.py 