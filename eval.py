import torch
from config import *
from utils import *
import torch_xla.core.xla_model as xm


def eval_model(models_info, test_path, orig_filenames, pred_filenames):
    device = xm.xla_device()
    print(device, "will be used.")

    print('Initializing models...')
    models = []
    for path, model_class, seq_len in models_info:
        model = model_class(seq_len)
        model.load_state_dict(torch.load(path))
        model.eval()
        models.append(model)

    print('Reading data file...')
    questions, decomps = read_input_file(test_path, format_decomps=False)

    print('Encoding questions...')
    inputs = model.encode(questions)
    inputs = mask(inputs, model.sep_id, model.mask_id)

    for i in range(len(models)):
        models[i] = models[i].to(device)

    predictions = [[] for _ in models]
    print('Processing...')
    with torch.no_grad():
        for i in range(0, len(questions), EVAL_BATCH_SIZE):
            print('Example', i)

            inp = inputs[i:i + EVAL_BATCH_SIZE].to(device)
            for m, model in enumerate(models):
                preds = model.clean(model.decode(inp, model(inp)))
                predictions[m] += preds

    print('Writing to files...')
    for i in range(len(models)):
        write_output_files(questions, decomps, predictions[i], orig_filenames[i], pred_filenames[i])

    print('Finished.')
