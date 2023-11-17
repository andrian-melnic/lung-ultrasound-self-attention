def fit_model(model, trainer, datamodule, checkpoint_path=None):
    print("\n\nTRAINING MODEL...")
    print('=' * 80 + "\n")
    if checkpoint_path:
        check_checkpoint(checkpoint_path)
        trainer.fit(model, datamodule, ckpt_path=checkpoint_path)
        
    else:
        print("Instantiating trainer without checkpoint...")
        trainer.fit(model, datamodule)
        
def test_model(model, trainer, datamodule, checkpoint_path=None):
    print("\n\nTESTING MODEL...")
    print('=' * 80 + "\n")
    if checkpoint_path:
        check_checkpoint(checkpoint_path)
        trainer.test(model, datamodule, ckpt_path=checkpoint_path)
    else:
        print("No checkpoint provided, testing from scratch...")
        trainer.test(model, datamodule)
        
        