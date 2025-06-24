from types import SimpleNamespace

def get_iseg_config():
    """
    Returns a config object for using iSeg inference in SVDiff training.
    Ignores dataset path args, only includes what iSeg actually uses at inference time.
    """
    return SimpleNamespace(
        batch_size=1,
        attention_layers_to_use=[
            'down_blocks[1].attentions[1].transformer_blocks[0].attn2',
            'down_blocks[2].attentions[0].transformer_blocks[0].attn2',
            'down_blocks[2].attentions[1].transformer_blocks[0].attn2',
            "up_blocks[1].attentions[0].transformer_blocks[0].attn2",
            "up_blocks[1].attentions[1].transformer_blocks[0].attn2",
            'up_blocks[1].attentions[2].transformer_blocks[0].attn1',
            "up_blocks[1].attentions[2].transformer_blocks[0].attn2",
            "up_blocks[2].attentions[0].transformer_blocks[0].attn2",
            "up_blocks[2].attentions[1].transformer_blocks[0].attn2",
            "up_blocks[3].attentions[1].transformer_blocks[0].attn1",
            'mid_block.attentions[0].transformer_blocks[0].attn1',
        ],
        gpu_id=0,
        train=False,
        text_prompt=None,
        output_dir="../outputs",

        # Optional or fixed training params
        optimizer="Adam",
        embeddings_file="",
        model_file=None,
        epochs=0,
        lr=0.05,
        train_mask_size=512,
        train_t=[50, 150],
        self_attention_loss_coef=1.0,
        sd_loss_coef=0.005,
        ent=0.1,

        # Test config
        masking="patched_masking",
        num_patchs_per_side=1,
        patch_size=512,
        patch_threshold=0.2,
        test_t=[100],
        test_mask_size=512,
        save_test_predictions=False,

        # Shell args
        save_file=None,
        rand_seed=1101,
        iter=10,
        enhanced=1.6,
        no_use_self_ers=True,
        no_use_cross_enh=True,
        no_use_cluster=True,

        # Manual minimal dataset-dependent stuff
        dataset_name="coco_object",
        num_class=80,
        cam_bg_thr=0.45,
        att_mean=True,
    )
