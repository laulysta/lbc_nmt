import argparse

import numpy
import theano
import cPickle as pkl
import time
import os

from nmt import (build_sampler_2, gen_sample_2, load_params,
                 init_params, init_tparams, prepare_data, greedy_decoding)

from data_iterator import TextIterator

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle
import ipdb

def main(model, src_dict, trg_dict, src, trg, multibleu, batch_size=60, pred_dir='', model_list=False):
    if pred_dir is not '' and not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    if model_list:
        model_list_file = model
        with open(model_list_file) as f:
            model=f.readline().strip()

    # load dictionaries and invert them
    worddicts = [None] * 2
    worddicts_r = [None] * 2
    for ii, dd in enumerate([src_dict, trg_dict]):
        with open(dd, 'rb') as f:
            worddicts[ii] = pkl.load(f)
        worddicts_r[ii] = dict()
        for kk, vv in worddicts[ii].iteritems():
            worddicts_r[ii][vv] = kk


    # load model options
    with open('%s.pkl' % model, 'rb') as f:
        options = pkl.load(f)

    trng = RandomStreams(options['trng'])
    use_noise = theano.shared(numpy.float32(0.))

    # allocate model parameters
    params = init_params(options)

    # load model parameters and set theano shared variables
    params = load_params(model, params)
    tparams = init_tparams(params)

    f_init_2, f_next_2 = build_sampler_2(tparams, options, trng, use_noise)

    iterator = TextIterator(src, trg, src_dict, trg_dict,
        n_words_source=options['n_words_src'],n_words_target=options['n_words'],
        batch_size=batch_size, maxlen=2000, shuffle=False, replace=False)

    if not model_list:
        try:
            valid_out, valid_bleu = greedy_decoding(options, trg, iterator, worddicts_r, tparams, prepare_data, gen_sample_2, f_init_2, f_next_2, trng,
            multibleu, fname=os.path.join(pred_dir, os.path.basename(model)[:-3]+'out'), maxlen=100, verbose=False)
        except:
            valid_out = ''
            valid_bleu = 0.0
        print valid_out, valid_bleu
    else:
        best_score = 0.
        best_model = ''
        with open(model_list_file) as f:
            for line in f:
                start = time.time()
                model = line.strip()
                if model == '':
                    continue
                params = load_params(model, params)
                for kk, pp in params.iteritems():
                    tparams[kk].set_value(params[kk])
                print model,
                try: 
                    valid_out, valid_bleu = greedy_decoding(options, trg, iterator, worddicts_r, tparams, prepare_data, gen_sample_2, f_init_2, f_next_2, trng,
                    multibleu, fname=os.path.join(pred_dir, os.path.basename(model)[:-3]+'out'), maxlen=100, verbose=False)
                except:
                    valid_out = ''
                    valid_bleu = 0.0
                print valid_out, valid_bleu,
                if valid_bleu > best_score:
                    best_score = valid_bleu
                    best_model = model
                end = time.time()
                print "Time: ", end-start
        print 'Best model: ', best_model
        print 'Best BLEU: ', best_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('src_dict', type=str)
    parser.add_argument('trg_dict', type=str)
    parser.add_argument('source', type=str)
    parser.add_argument('trg', type=str)
    parser.add_argument('multibleu', type=str)
    parser.add_argument('batch_size', type=int, default=60)
    parser.add_argument('pred_dir', type=str)
    parser.add_argument('--model_list', action="store_true", default=False)

    args = parser.parse_args()

    main(args.model, args.src_dict, args.trg_dict, args.source,
         args.trg, args.multibleu, args.batch_size, 
         args.pred_dir, args.model_list)
