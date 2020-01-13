import string
import random
import torch
from torch import nn
from torchtext import data
import sacrebleu
from sacremoses import MosesDetokenizer
from configuration import cfg
from reader import src_tokenizer

detokenizer = MosesDetokenizer(lang=cfg.tgt_lang)


def convert_target_batch_back(btch, TGT):
    """
    :param btch: seq_length, batch_size
    This function must look at attention scores for <unk> tokens and replace them with words from input if possible
    :return:
    """
    non_desired__ids = [TGT.vocab.stoi[cfg.pad_token], TGT.vocab.stoi[cfg.bos_token]]
    eos = TGT.vocab.stoi[cfg.eos_token]
    tgt_vocab_size = len(TGT.vocab.itos)
    s_len, batch_size = btch.size()
    result = []
    for b in range(batch_size):
        tmp_res = []
        for w in range(s_len):
            itm = btch[w, b]
            # p_gen = p_gens[w, b].item() if p_gens is not None else 1.0
            if itm == eos:
                break
            if itm not in non_desired__ids:
                int_itm = int(itm)
                if int_itm < tgt_vocab_size:
                    cvrted = TGT.vocab.itos[int_itm]
                    # assert p_gen > 0.5, "P_gen is {:.3f}".format(p_gen)  # To make sure there is no bug in training
                else:
                    # assert int_itm != tgt_vocab_size
                    ptr_id = int_itm - tgt_vocab_size - 1
                    cvrted = "<src>_{}".format(ptr_id)
                tmp_res.append(cvrted)
                """
                if p_gen > 0.5:
                    cvrted = TGT.vocab.itos[int(itm)]
                else:
                    cvrted = cfg.unk_token
                """
        result.append(" ".join(tmp_res))
    return result
    # return [" ".join([TGT.vocab.itos[int(btch[w, b])]
    #                  for w in range(s_len) if btch[w, b] not in non_desired__ids]) for b in range(batch_size)]


def recover_copy_tokens(decoded: str, source: str):
    source_tokens = source.split()
    decoded_tokens = decoded.split()
    # TODO try lexical translation first
    try:
        result = [token if "<src>" not in token else source_tokens[int(token.split("_")[-1])] for token in decoded_tokens]
        return " ".join(result)
    except IndexError:
        return decoded


def postprocess_decoded(decoded_sentence, input_sentence, attention_scores):
    source_sentence_tokenized = src_tokenizer(input_sentence)
    max_input_sentence_length = len(source_sentence_tokenized)
    max_decode_length = attention_scores.size(0)
    decoded_sentence_tokens = decoded_sentence.split()
    result = []
    for tgt_id, tgt_token in enumerate(decoded_sentence_tokens):
        if tgt_token == cfg.unk_token and tgt_id < max_decode_length:
            input_sentence_id = int(attention_scores[tgt_id].item())
            if input_sentence_id < max_input_sentence_length:
                lex = source_sentence_tokenized[input_sentence_id]
                # TODO try lexical translation first
                result.append(lex)
                continue
        result.append(tgt_token)
    # Naive detokenization
    # return "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in result]).strip()
    return detokenizer.detokenize(result)


def evaluate(data_iter: data.BucketIterator, TGT: data.field, model: nn.Module,
             src_file: str, gold_tgt_file: str, eph: str):
    print("Evaluation ....")
    model.eval()
    src_originals = iter(open(src_file, "r"))
    tgt_originals = iter(open(gold_tgt_file, "r"))
    random_sample_created = False
    with torch.no_grad():
        lall_valid = 0.0
        lcount_valid = 0.0
        all_bleu_score = 0.0
        sent_count = 0.0
        for valid_instance in data_iter:
            pred, max_attention_idcs, lss, _, n_tokens = model(valid_instance.src, valid_instance.trg, test_mode=True)
            lall_valid += lss.item()
            lcount_valid += n_tokens
            for d_id, (decoded, model_expected) in enumerate(zip(
                    convert_target_batch_back(pred, TGT), convert_target_batch_back(valid_instance.trg[0], TGT))):
                source_sentence = next(src_originals).strip()
                reference_sentence = next(tgt_originals).strip()
                if bool(cfg.lowercase_data):
                    source_sentence = source_sentence.lower()
                    reference_sentence = reference_sentence.lower()
                if "<src>" in model_expected:
                    raise ValueError("Copy mechanism is not possible to have kicked in the reference.")
                if "<src>" in decoded:
                    decoded = recover_copy_tokens(decoded, source_sentence)
                decoded = postprocess_decoded(decoded, source_sentence, max_attention_idcs.select(1, d_id))
                if bool(cfg.dataset_is_in_bpe):
                    decoded = decoded.replace("@@ ", "")
                    reference_sentence = reference_sentence.replace("@@ ", "")
                all_bleu_score += sacrebleu.corpus_bleu([decoded], [[reference_sentence]]).score
                sent_count += 1.0
                if not random_sample_created and random.random() < 0.01:
                    random_sample_created = True
                    print("Source Sent': {}\nSample Pred : {}\nModel Expc'd: {}\nSample Act'l: {}".format(
                        source_sentence, decoded, model_expected, reference_sentence))
        # valid_instance = next(iter(val_iter))
        # pred, _, _, _ = model(valid_instance.src, valid_instance.trg)
        # cpreds = convert_target_batch_back(pred)
        # cactuals = convert_target_batch_back(valid_instance.trg[0])
        # ind = random.randint(0, len(cpreds)-1)
        average_loss = lall_valid/max(lcount_valid, 1)
        average_bleu = all_bleu_score / max(sent_count, 1)
        print("E {} ::: Average Loss {:.3f} ::: Average BleuP1 {:.3f}".format(eph, average_loss, average_bleu))
    model.train()
    return average_loss, average_bleu
