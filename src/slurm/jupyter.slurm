#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00        
#SBATCH --partition=a100
 
# Charger votre environnement
source .venv/bin/activate
 
# Trouver le nom du nœud sur lequel le job tourne
NODE_NAME=$(hostname)
PORT=8888
 
echo "------------------------------------------------------"
echo "Le serveur Jupyter tourne sur le nœud : $NODE_NAME"
echo "Port utilisé : $PORT"
echo "------------------------------------------------------"
 
# Lancer Jupyter
jupyter notebook --no-browser --port=$PORT --ip=0.0.0.0
