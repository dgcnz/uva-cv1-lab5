
run-tests:
	pytest test/

sweep-lenet5-cifar100:
	wandb sweep --project sweep-lenet5-cifar100 sweep_lenet5.yaml

sweep-twolayer-cifar100:
	wandb sweep --project sweep-twolayer-cifar100 sweep_twolayer.yaml


sweep-lenet5-finetune:
	wandb sweep --project sweep-lenet5-finetune sweep_lenet5_finetune.yaml

sweep-lenet5deep-finetune:
	wandb sweep --project sweep-lenet5deep-finetune sweep_lenet5deep_finetune.yaml