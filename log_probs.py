import argparse

import numpy
import theano
import cPickle as pkl

from nmt import (load_params, init_params, init_tparams,
    build_model, pred_probs, prepare_data)

from data_iterator import TextIterator

profile = False

def main(model, dictionary, dictionary_target, source, target, outfile, wordbyword):

    # load model model_options
    with open('%s.pkl' % model, 'rb') as f:
        options = pkl.load(f)
    """
    # load source dictionary and invert
    with open(dictionary, 'rb') as f:
        word_dict = pkl.load(f)
    word_idict = dict()
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # load target dictionary and invert
    with open(dictionary_target, 'rb') as f:
        word_dict_trg = pkl.load(f)
    word_idict_trg = dict()
    for kk, vv in word_dict_trg.iteritems():
        word_idict_trg[vv] = kk
    word_idict_trg[0] = '<eos>'
    word_idict_trg[1] = 'UNK'
    """
    valid_noshuf = TextIterator(source, target,
                         dictionary, dictionary_target,
                         n_words_source=options['n_words_src'], n_words_target=options['n_words'],
                         batch_size=options['valid_batch_size'], maxlen=2000, shuffle=False)

    # allocate model parameters
    params = init_params(options)

    # load model parameters and set theano shared variables
    params = load_params(model, params)
    tparams = init_tparams(params)

    trng, use_noise, \
        x, x_mask, y, y_mask, \
        opt_ret, \
        cost, cost_ = \
        build_model(tparams, options)

    inps = [x, x_mask, y, y_mask]

    if wordbyword:
        f_log_probs = theano.function(inps, cost_, profile=profile)
        valid_errs = pred_probs(f_log_probs, prepare_data, options, valid_noshuf, verbose=True, as_list=True)
        with open(outfile, 'wb') as f:
            pkl.dump(valid_errs, f, pkl.HIGHEST_PROTOCOL)
    else:
        f_log_probs = theano.function(inps, cost, profile=profile)
        valid_errs = pred_probs(f_log_probs, prepare_data, options, valid_noshuf, verbose=True)
        numpy.save(outfile, valid_errs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('dictionary', type=str)
    parser.add_argument('dictionary_target', type=str)
    parser.add_argument('source', type=str)
    parser.add_argument('target', type=str)
    parser.add_argument('outfile', type=str)
    parser.add_argument('--wordbyword', action='store_true', default=False)

    args = parser.parse_args()

    main(args.model, args.dictionary, args.dictionary_target,
         args.source, args.target, args.outfile, args.wordbyword)

