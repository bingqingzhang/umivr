from retrievalmodel.ivr_base import \
    ivr_heuristic_multiturn, ivr_auto_multiturn, ua_multiturn

dialogue_registry = {
    "heuristic_ivr_multiturn": ivr_heuristic_multiturn,
    "auto_ivr_multiturn": ivr_auto_multiturn,
    "ua_multiturn": ua_multiturn,
}

def do_dialogue_retrieval(dialogue_type, **kwargs):
    func = dialogue_registry.get(dialogue_type, None)
    if func:
        return func(**kwargs)
    else:
        raise ValueError(f"Dialogue retrieval function {dialogue_type} not found")
    
def assemble_parameters_ivr(type, test_video_dataloader, test_text_dataloader, model, config, retrieval_tokenizer, args, processor, generation_tokenizer):
    assmebled_params = {}
    if type == "heuristic_ivr_multiturn":
        assmebled_params.update({
            "test_video_dataloader": test_video_dataloader,
            "test_text_dataloader": test_text_dataloader,
            "config": config,
            "model": model,
            "retrieval_tokenizer": retrieval_tokenizer,
            "generation_tokenizer": generation_tokenizer,
            "device": args.device,
            "preprocess": processor['video']
        })
    elif type == "auto_ivr_multiturn":
        assmebled_params.update({
            "test_video_dataloader": test_video_dataloader,
            "test_text_dataloader": test_text_dataloader,
            "config": config,
            "model": model,
            "retrieval_tokenizer": retrieval_tokenizer,
            "generation_tokenizer": generation_tokenizer,
            "device": args.device,
            "preprocess": processor['video']
        })
    elif type == "ua_multiturn":
        assmebled_params.update({
            "test_video_dataloader": test_video_dataloader,
            "test_text_dataloader": test_text_dataloader,
            "config": config,
            "model": model,
            "retrieval_tokenizer": retrieval_tokenizer,
            "generation_tokenizer": generation_tokenizer,
            "device": args.device,
            "preprocess": processor['video']
        })
    else:
        raise ValueError(f"Dialogue retrieval function {type} not found")
    return assmebled_params