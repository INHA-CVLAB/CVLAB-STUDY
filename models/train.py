import argparse
from data_loader.mnist import load_mnist
import utils

def get_argument_parser(model_name):
    '''
    Argument parser which returns the options which the user inputted.
    '''
    weights_path = './saved_model/{model_name}.h5'
    image_path = '../figures/{model_name}.png'
    plot_path = '../figures/{model_name}_plot.png'

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',
                        help = 'How many epochs you need to run (default: 10)',
                        type = int, default = 10)
    parser.add_argument('--batch_size',
                        help = 'The number of images in a batch (default: 64)',
                        type = int, default = 64)
    parser.add_argument('--path_for_weights',
                        help = 'The path from where the weights will be saved or loaded \
                                (default: {weights_path})',
                        type = str, default = weights_path)
    parser.add_argument('--path_for_image',
                        help = 'The path from where the model image will be saved \
                                (default: {image_path})',
                        type = str, default = image_path)
    parser.add_argument('--path_for_plot',
                        help = 'The path from where the training progress will be plotted \
                                (default: {plot_path})',
                        type = str, default = plot_path)
    parser.add_argument('--data_augmentation',
                        help = '0: No, 1: Yes (default: 1)',
                        type = int, default = 1)
    parser.add_argument('--save_model_and_weights',
                        help = '0: No, 1: Yes (default: 1)',
                        type = int, default = 1)
    parser.add_argument('--load_weights',
                        help = '0: No, 1: Yes (default: 0)',
                        type = int, default = 0)
    parser.add_argument('--plot_training_progress',
                        help = '0: No, 1: Yes (default: 1)',
                        type = int, default = 1)
    parser.add_argument('--save_model_to_image',
                        help = '0: No, 1: Yes (default: 1)',
                        type = int, default = 1)
    args = parser.parse_args()

    return args

def train(model, model_name):
    # load all arguments
    args = get_argument_parser(model_name)

    training_data, validation_data, test_data = load_mnist()
    print('[data loaded]')

    # build and compile the model
    model.compile()
    print('[model built]')

    # save the model architecture to an image file
    if args.save_model_to_image:
        model.save_model_as_image(args.path_for_image)
        print('[model image saved as {args.path_for_image}]')

    # TODO: Need to revise to automatically check pretrained model.
    # load pretrained weights
    if args.load_weights:
        model.load_weights(args.path_for_weights)
        print('[weights loaded from {args.path_for_weights}]')

    # train the model
    hist = None
    if args.data_augmentation:
        hist = model.fit_generator(training_data, validation_data,
                                   epochs = args.epochs, batch_size = args.batch_size)
        print('[trained with augmented images]')
    else:
        hist = model.fit(training_data, validation_data,
                         epochs = args.epochs, batch_size = args.batch_size)
        print('[trained without augmented images]')

    # save the training progress to an image file
    if args.plot_training_progress:
        utils.plot(history = hist, path = args.path_for_plot, title = model_name)
        print('[training progress saved as {args.path_for_plot}]')

    # save the model and trained weights in the configured path
    if args.save_model_and_weights:
        model.save(args.path_for_weights)
        print('[Model and trained weights saved in {args.path_for_weights}]')

    # evaluate the model with the test dataset
    loss_and_metrics = model.evaluate(test_data, batch_size = args.batch_size)
    print('[Evaluation on the test dataset]\n', loss_and_metrics, '\n')

