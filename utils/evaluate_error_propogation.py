import sys
import json


def evaluate_value(value):
    myvalue = json.loads(value.rstrip())
    decision_errors = len(myvalue)
    first_error = -1
    total_loss = 0
    error_locations = 0
    av_error_location = 0
    if decision_errors > 0:
        first_error = myvalue[0][0]
        total_loss = myvalue[-1][1]
        for val in myvalue:
            error_locations += val[0]
        av_error_location = float(error_locations / decision_errors)

    return decision_errors, total_loss, first_error, av_error_location


def get_average_of_list(mylist):
    mytotal = 0
    for numb in mylist:
        mytotal += numb
    return float(mytotal / len(mylist))


def get_evaluation_information_from_file(myinput_file):
    myin = open(myinput_file, 'r')
    current_sentence = '-1'
    completely_correct_sentences = 0
    sentences = 0
    corpus_loss = 0
    corpus_dec_errors = 0
    first_errors = []
    average_errors = []
    hard_error_prop_s = []
    fixes = 0
    total_soft_error_propagation = 0
    total_new_errors = 0
    soft_error_propagation = 0
    soft_error_props = []
    new_errors_caused = []
    independent_errors = []
    total_errors = []
    overall_loss = []
    losses_per_fix = []
    losses_per_derr = []
    for line in myin:
        parts = line.split('\t')
        # get all new stuff, start counting
        if parts[0] != current_sentence:
            # new sentence: we're now done measuring soft error propagation


            # simlpified assumption
            if current_sentence != '-1':
                soft_error_propagation = dec_error - fixes
                if soft_error_propagation > 0:
                    soft_error_props.append(soft_error_propagation)
                    total_soft_error_propagation += soft_error_propagation
                    soft_error_propagation = 0
                elif soft_error_propagation < 0:
                    total_new_errors += soft_error_propagation
                independent_error = dec_error - soft_error_propagation
                independent_errors.append(independent_error)
                total_errors.append(dec_error)
                overall_loss.append(total_loss)
                if fixes > 0:
                    loss_per_fix = total_loss / fixes
                    error_per_fix = dec_error / fixes
                    losses_per_fix.append(loss_per_fix)
                    losses_per_derr.append(error_per_fix)
            dec_error, total_loss, first_error, av_error = evaluate_value(parts[1])
            sentences += 1
            sent_loss = 0
            sent_dec_errors = 0

            if dec_error == 0:
                completely_correct_sentences += 1
            else:
                corpus_loss += total_loss
                corpus_dec_errors += dec_error
                first_errors.append(first_error)
                average_errors.append(av_error)
                hard_error_prop = total_loss - dec_error
                hard_error_prop_s.append(hard_error_prop)
                sent_loss = total_loss
                sent_dec_errors = dec_error
            current_sentence = parts[0]
            fixes = 0
        else:
            current_dec_error, current_loss, cfirst_error, cav_error = evaluate_value(parts[1])
            fixes += 1
            loss_reduction = sent_loss - current_loss
            sent_loss = current_loss
            error_reduction = sent_dec_errors - current_dec_error
            sent_dec_errors = current_dec_error
            if error_reduction > 0:
                soft_error_propagation += error_reduction - 1
            elif not current_dec_error == 0:
                new_errors_caused.append(error_reduction)

    print('Sentences: ' + str(sentences))
    print('Completely correct sentences: ' + str(completely_correct_sentences))
    print('Corpus loss: ' + str(corpus_loss))
    print('Corpus decision errors: ' + str(corpus_dec_errors))
    print('Hard error propagation: ' + str(sum(hard_error_prop_s)))
    print('Soft error propagation: ' + str(total_soft_error_propagation))
    print('Average first error location: ' + str(get_average_of_list(first_errors)))
    print('Average error location: ' + str(get_average_of_list(average_errors)))
    print('Average loss increase per fix: ' + str(get_average_of_list(losses_per_fix)))
    print('Average error reduction per fix: ' + str(get_average_of_list(losses_per_derr)))
    print('Average loss per sentence (all sentences): ' + str(float(corpus_loss / sentences)))
    print('Average loss per sentence (erroneous sentences): ' + str(
        float(corpus_loss / (sentences - completely_correct_sentences))))
    print('Average errors per sentence (all sentences): ' + str(float(corpus_dec_errors / sentences)))
    print('Average errors per sentence (erroneous sentences): ' + str(
        float(corpus_dec_errors / (sentences - completely_correct_sentences))))
    print('Percentage of hard error propagation errors: ' + str(float(sum(hard_error_prop_s) / corpus_loss)))
    print(
        'Percentage of soft error propagation errors: ' + str(float(total_soft_error_propagation / corpus_dec_errors)))
    print('Percentage of independent errors: ' + str(
        float((corpus_dec_errors - total_soft_error_propagation) / corpus_dec_errors)))
    print('Number of new errors through fixes: ' + str(total_new_errors))


###for subsequent errors: look at differences



def main():
    myargs = sys.argv
    if len(myargs) < 2:
        print('Error: you need an input document')
    else:
        get_evaluation_information_from_file(myargs[1])


if __name__ == '__main__':
    main()