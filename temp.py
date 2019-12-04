def predict(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double predictuation (matched, mis-matched)
    predict_task_names =  (args.task_name,)
    predict_outputs_dirs = (args.output_dir,)

    results = {}
    for predict_task, predict_output_dir in zip(predict_task_names, predict_outputs_dirs):
        predict_dataset = load_and_cache_examples(args, predict_task, tokenizer, predictuate=True)

        if not os.path.exists(predict_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(predict_output_dir)

        args.predict_batch_size = args.per_gpu_predict_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        predict_sampler = SequentialSampler(predict_dataset) if args.local_rank == -1 else DistributedSampler(predict_dataset)
        predict_dataloader = DataLoader(predict_dataset, sampler=predict_sampler, batch_size=args.predict_batch_size)

        # multi-gpu predict
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # predict!
        logger.info("***** Running predictuation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(predict_dataset))
        logger.info("  Batch size = %d", args.predict_batch_size)
        predict_loss = 0.0
        nb_predict_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(predict_dataloader, desc="predictuating"):
            model.predict()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_predict_loss, logits = outputs[:2]

                predict_loss += tmp_predict_loss.mean().item()
            nb_predict_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        predict_loss = predict_loss / nb_predict_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(predict_task, preds, out_label_ids)
        results.update(result)

        output_predict_file = os.path.join(predict_output_dir, prefix, "predict_results.txt")
        with open(output_predict_file, "w") as writer:
            logger.info("***** predict results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results
