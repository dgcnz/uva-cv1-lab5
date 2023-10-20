
run-tests:
	pytest test/

sweep-lenet5-cifar100:
	wandb sweep --project sweep-lenet5-cifar100 sweep_lenet5.yaml

sweep-twolayer-cifar100:
	wandb sweep --project sweep-twolayer-cifar100 sweep_twolayer.yaml
