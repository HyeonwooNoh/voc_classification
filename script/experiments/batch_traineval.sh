# Compare different batch size
python -m src.traineval.traineval --model resnet18 --batch_size 16\
	--num_epochs 50 --initial_learning_rate 0.1 --momentum 0.9 --seed 111\
	--save_dir 'results/batch_traineval/resnet18_16_50_0.1_0.9_111'
python -m src.traineval.traineval --model resnet18 --batch_size 32\
	--num_epochs 50 --initial_learning_rate 0.1 --momentum 0.9 --seed 111\
	--save_dir 'results/batch_traineval/resnet18_32_50_0.1_0.9_111'
python -m src.traineval.traineval --model resnet18 --batch_size 64\
	--num_epochs 50 --initial_learning_rate 0.1 --momentum 0.9 --seed 111\
	--save_dir 'results/batch_traineval/resnet18_64_50_0.1_0.9_111'
python -m src.traineval.traineval --model resnet18 --batch_size 128\
	--num_epochs 50 --initial_learning_rate 0.1 --momentum 0.9 --seed 111\
	--save_dir 'results/batch_traineval/resnet18_128_50_0.1_0.9_111'
python -m src.traineval.traineval --model resnet18 --batch_size 256\
	--num_epochs 50 --initial_learning_rate 0.1 --momentum 0.9 --seed 111\
	--save_dir 'results/batch_traineval/resnet18_256_50_0.1_0.9_111'

# Compare different learning rates
python -m src.traineval.traineval --model resnet18 --batch_size 128\
	--num_epochs 50 --initial_learning_rate 1.0 --momentum 0.9 --seed 111\
	--save_dir 'results/batch_traineval/resnet18_128_50_1.0_0.9_111'
python -m src.traineval.traineval --model resnet18 --batch_size 128\
	--num_epochs 50 --initial_learning_rate 0.5 --momentum 0.9 --seed 111\
	--save_dir 'results/batch_traineval/resnet18_128_50_0.5_0.9_111'
python -m src.traineval.traineval --model resnet18 --batch_size 128\
	--num_epochs 50 --initial_learning_rate 0.05 --momentum 0.9 --seed 111\
	--save_dir 'results/batch_traineval/resnet18_128_50_0.05_0.9_111'
python -m src.traineval.traineval --model resnet18 --batch_size 128\
	--num_epochs 50 --initial_learning_rate 0.01 --momentum 0.9 --seed 111\
	--save_dir 'results/batch_traineval/resnet18_128_50_0.01_0.9_111'

# Compare different models
python -m src.traineval.traineval --model resnet18 --batch_size 28\
	--num_epochs 50 --initial_learning_rate 0.1 --momentum 0.9 --seed 111\
	--save_dir 'results/batch_traineval/resnet18_28_50_0.1_0.9_111'
python -m src.traineval.traineval --model resnet34 --batch_size 28\
	--num_epochs 50 --initial_learning_rate 0.1 --momentum 0.9 --seed 111\
	--save_dir 'results/batch_traineval/resnet34_28_50_0.1_0.9_111'
python -m src.traineval.traineval --model resnet50 --batch_size 28\
	--num_epochs 50 --initial_learning_rate 0.1 --momentum 0.9 --seed 111\
	--save_dir 'results/batch_traineval/resnet50_28_50_0.1_0.9_111'
python -m src.traineval.traineval --model resnet101 --batch_size 28\
	--num_epochs 50 --initial_learning_rate 0.1 --momentum 0.9 --seed 111\
	--save_dir 'results/batch_traineval/resnet100_28_50_0.1_0.9_111'
python -m src.traineval.traineval --model resnet152 --batch_size 28\
	--num_epochs 50 --initial_learning_rate 0.1 --momentum 0.9 --seed 111\
	--save_dir 'results/batch_traineval/resnet152_28_50_0.1_0.9_111'

# Compare different momentums
python -m src.traineval.traineval --model resnet18 --batch_size 128\
	--num_epochs 50 --initial_learning_rate 0.1 --momentum 0.7 --seed 111\
	--save_dir 'results/batch_traineval/resnet18_128_50_0.1_0.7_111'
python -m src.traineval.traineval --model resnet18 --batch_size 128\
	--num_epochs 50 --initial_learning_rate 0.1 --momentum 0.5 --seed 111\
	--save_dir 'results/batch_traineval/resnet18_128_50_0.1_0.5_111'
python -m src.traineval.traineval --model resnet18 --batch_size 128\
	--num_epochs 50 --initial_learning_rate 0.1 --momentum 0.3 --seed 111\
	--save_dir 'results/batch_traineval/resnet18_128_50_0.1_0.3_111'

# Using weight decay
python -m src.traineval.traineval --model resnet18 --batch_size 128\
	--num_epochs 200 --initial_learning_rate 0.1 --momentum 0.9 --seed 111\
	--learning_rate_decay_epoch 50\
	--save_dir 'results/batch_traineval/resnet18_128_200_0.1_0.9_111_50'
python -m src.traineval.traineval --model resnet18 --batch_size 128\
	--num_epochs 200 --initial_learning_rate 0.1 --momentum 0.9 --seed 111\
	--learning_rate_decay_epoch 100\
	--save_dir 'results/batch_traineval/resnet18_128_200_0.1_0.9_111_100'

# Compare different seeds
python -m src.traineval.traineval --model resnet18 --batch_size 128\
	--num_epochs 50 --initial_learning_rate 0.1 --momentum 0.9 --seed 123\
	--save_dir 'results/batch_traineval/resnet18_128_50_0.1_0.9_123'
python -m src.traineval.traineval --model resnet18 --batch_size 128\
	--num_epochs 50 --initial_learning_rate 0.1 --momentum 0.9 --seed 234\
	--save_dir 'results/batch_traineval/resnet18_128_50_0.1_0.9_234'
python -m src.traineval.traineval --model resnet18 --batch_size 128\
	--num_epochs 50 --initial_learning_rate 0.1 --momentum 0.9 --seed 345\
	--save_dir 'results/batch_traineval/resnet18_128_50_0.1_0.9_345'
python -m src.traineval.traineval --model resnet18 --batch_size 128\
	--num_epochs 50 --initial_learning_rate 0.1 --momentum 0.9 --seed 456\
	--save_dir 'results/batch_traineval/resnet18_128_50_0.1_0.9_456'

