#!/usr/bin/zsh

#conda init zsh

#conda activate jax_qdots
# Run the first program


python run_fully_adaptive.py

# Check if the first program finished successfully
if [ $? -ne 0 ]; then
  echo "First program failed"
  exit 1
fi

# Run the second program
python run_t_opt.py

# Check if the second program finished successfully
if [ $? -ne 0 ]; then
  echo "Second program failed"
  exit 1
fi

# Run the third program
python run_trace_fully_adaptive.py

# Check if the third program finished successfully
if [ $? -ne 0 ]; then
  echo "Third program failed"
  exit 1
fi



# Run the third program
python run_vanilla.py

# Check if the third program finished successfully
if [ $? -ne 0 ]; then
  echo "Fourth program failed"
  exit 1
fi



echo "All programs executed successfully"