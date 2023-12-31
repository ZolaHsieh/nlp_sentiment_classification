import torch
import argparse

from torch import nn, optim
from model_training import train
from transformers import AutoTokenizer
from timeit import default_timer as timer
from utils import plot_loss_curves, save_model
from torch.utils.data import TensorDataset, DataLoader
from data_preprocess import load_imdb_data, preprocess_text, get_tknr, create_vocab, word2id_padding, get_glove_weight
from models import LSTMModel, LSTMGloveModel, BiLSTMModel, DistilBertModel

torch.manual_seed(1126)
torch.cuda.manual_seed(1126)

emb_dim = 128
hidden_dim = 128
n_layers = 2
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--m', dest='model_type', default='lstm', help='set model for training (default: lstm)')
    args.add_argument('--e', dest='epochs', default=2, type=int, help='training epochs')
    args = args.parse_args()
    
    epochs = args.epochs
    print(f"Device:{device}, model:{args.model_type}, training epochs: {epochs}")

    ## get data
    print("=== Load imdb data === ")
    train_data, test_data = load_imdb_data()

    ## data preprocess
    # train_data = train_data.map(preprocess_text)
    # test_data = test_data.map(preprocess_text)

    if args.model_type != "distilbert":
        ## vocab & tokenizer
        print("=== create tokeizer, vocab=== ")

        tokenizer = get_tknr()
        vocab = create_vocab(tokenizer, train_data)

        ## setup dataloader
        print("=== create data loader=== ")
        train_x, train_y = word2id_padding(vocab, tokenizer, train_data, max_len=256)
        test_x, test_y = word2id_padding(vocab, tokenizer, test_data, max_len=256)

        train_set = TensorDataset(train_x, train_y)
        test_set = TensorDataset(test_x, test_y)

        train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(test_set, batch_size=32)
        
        ## create model
        print(f"=== create model: {args.model_type} ===")
        if args.model_type=="lstm":
            model = LSTMModel(vocab_len = len(vocab),
                              emb_dim = emb_dim,
                              hidden_dim = hidden_dim,
                              n_layers = n_layers,
                              out_len = 2)
            
        elif args.model_type=="glove-lstm":
             # use glove.6B.100d and the emb_dim = 100
             weight = get_glove_weight(vocab = vocab, emb_dim = 100)
             model = LSTMGloveModel(emb_dim = 100,
                                    hidden_dim = hidden_dim,
                                    n_layers = n_layers,
                                    out_len = 2, 
                                    weight = weight)


        elif args.model_type=="bi-lstm":
                model = BiLSTMModel(vocab_len = len(vocab),
                                    emb_dim = emb_dim,
                                    hidden_dim = hidden_dim,
                                    n_layers = n_layers,
                                    out_len = 2)
                
        ## create loss func & optimizer
        print("=== create loss func & optimizer ===")
        loss_fn = nn.CrossEntropyLoss()
        optimizer= optim.Adam(model.parameters())

        ## model training
        print("=== Model training ===")
        start_time = timer()
        model_results = train(model=model,
                            train_dataloader=train_dataloader,
                            test_dataloader=test_dataloader,
                            optimizer=optimizer,
                            loss_fn = loss_fn,
                            epochs=epochs)
        end_time = timer()
        print(f"Total training time: {end_time-start_time:.3f} seconds")
    
    else: ## distilbert
        # get tokenizer
        print("=== create tokeizer === ")
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

        print("=== create data loader=== ")
        train_data = train_data.map(lambda e: tokenizer(e['text'], padding="max_length", max_length=256, truncation=True),
                                    batched=True)

        test_data = test_data.map(lambda e: tokenizer(e['text'], padding="max_length", max_length=256, truncation=True),
                                    batched=True)
        
        train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=32)

        print(f"=== create model: {args.model_type} ===")
        model = DistilBertModel(num_labels=2)
        for param in model.base_model.parameters():
            param.requires_grad = False

        ## create loss func & optimizer
        print("=== create loss func & optimizer ===")
        loss_fn = nn.CrossEntropyLoss()
        optimizer= optim.Adam(model.parameters())

        ## model training
        print("=== Model training ===")
        start_time = timer()
        model_results = train(model=model,
                              train_dataloader=train_dataloader,
                              test_dataloader=test_dataloader,
                              optimizer=optimizer,
                              loss_fn = loss_fn,
                              epochs=epochs,
                              bert = True)
        end_time = timer()
        print(f"Total training time: {end_time-start_time:.3f} seconds")


    ## print model performance
    print("=== models performance ===")
    for k, v in model_results.items():
        print(f"{k}: {v[-1]:.3f}")

    ## plot model result
    print("=== plot model loss & accuracy curve ===")
    plot_loss_curves(f"{args.model_type.capitalize()} Result" ,model_results)

    ## model to file
    print("=== model to file ===")
    save_model(model=model, target_dir="models", model_name=f"{args.model_type}_sentiment_clf.pth")

