"""
DESCRIPTION: operations to train a GAIN model.
AUTHOR: Pablo Ferri
DATE: 20/08/2023
"""

# MODULES IMPORT
import matplotlib.pyplot as mpl
from torch import no_grad
from torch.cuda import is_available
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from Imputation.GAIN.discgen import DiscriminatorCustom, GeneratorCustom
from Imputation.GAIN.lossesgain import loss_discriminator, loss_generator


# ADVERSARIAL IMPUTER TRAINER
class AdversarialImputerTrainer:
    # CLASS ATTRIBUTES
    _number_decimals2display = 4
    _linewidth = 3
    _marker = 'o'
    _markersize = 15
    _markeredgecolor = 'black'
    _markeredgewidth = 1

    # INITIALIZATION
    def __init__(self, *, discriminator: DiscriminatorCustom, generator: GeneratorCustom, loader_train: DataLoader,
                 loader_eval: DataLoader, learning_rate_discriminator: float = 0.0001,
                 learning_rate_generator: float = 0.0001, number_epochs: int = 100) -> None:
        # Initialization
        # optimizers
        optimizer_discriminator = AdamW(params=discriminator.parameters(), lr=learning_rate_discriminator)
        optimizer_generator = AdamW(params=generator.parameters(), lr=learning_rate_generator)

        # loss functions
        lossfun_discriminator = loss_discriminator
        lossfun_generator = loss_generator

        # loss storage variables
        # discriminator
        loss_epochs_discriminator_train = []
        loss_epochs_discriminator_eval = []
        # generator
        loss_epochs_generator_train = []
        loss_epochs_generator_classification_train = []
        loss_epochs_generator_reconstruction_train = []
        loss_epochs_generator_eval = []
        loss_epochs_generator_classification_eval = []
        loss_epochs_generator_reconstruction_eval = []

        # cuda allocation
        # flag
        use_gpu = is_available()
        # model allocation
        if use_gpu:
            discriminator.cuda()
            generator.cuda()
            print('Adversarial training will be carried out on GPU.')

        # Attributes assignation
        # models
        self._discriminator = discriminator
        self._generator = generator
        # data
        self._loader_train = loader_train
        self._loader_eval = loader_eval
        # use gpu
        self._use_gpu = use_gpu
        # optimizers
        self._optimizer_discriminator = optimizer_discriminator
        self._optimizer_generator = optimizer_generator
        # number epochs
        self._number_epochs = number_epochs
        # loss functions
        self._lossfun_discriminator = lossfun_discriminator
        self._lossfun_generator = lossfun_generator
        # loss values
        # discriminator
        self._loss_epochs_discriminator_train = loss_epochs_discriminator_train
        self._loss_epochs_discriminator_eval = loss_epochs_discriminator_eval
        # generator
        self._loss_epochs_generator_train = loss_epochs_generator_train
        self._loss_epochs_generator_classification_train = loss_epochs_generator_classification_train
        self._loss_epochs_generator_reconstruction_train = loss_epochs_generator_reconstruction_train
        self._loss_epochs_generator_eval = loss_epochs_generator_eval
        self._loss_epochs_generator_classification_eval = loss_epochs_generator_classification_eval
        self._loss_epochs_generator_reconstruction_eval = loss_epochs_generator_reconstruction_eval

    # EXTERNAL ATTRIBUTE ACCESS AND EDITION CONTROL
    @property
    def discriminator(self):
        return self._discriminator

    @property
    def generator(self):
        return self._generator

    @property
    def loss_epochs_discriminator_train(self):
        return self._loss_epochs_discriminator_train

    @property
    def loss_epochs_discriminator_eval(self):
        return self._loss_epochs_discriminator_eval

    @property
    def loss_epochs_generator_train(self):
        return self._loss_epochs_generator_train

    @property
    def loss_epochs_generator_classification_train(self):
        return self._loss_epochs_generator_classification_train

    @property
    def loss_epochs_generator_reconstruction_train(self):
        return self._loss_epochs_generator_reconstruction_train

    @property
    def loss_epochs_generator_eval(self):
        return self._loss_epochs_generator_eval

    @property
    def loss_epochs_generator_classification_eval(self):
        return self._loss_epochs_generator_classification_eval

    @property
    def loss_epochs_generator_reconstruction_eval(self):
        return self._loss_epochs_generator_reconstruction_eval

    # TRAINING
    def train(self):

        # Iteration across epochs
        for epoch in tqdm(range(self._number_epochs), colour='green'):

            # Memory allocation
            # discriminator
            loss_batches_discriminator_train = []
            loss_batches_discriminator_eval = []
            # generator
            loss_batches_generator_train = []
            loss_batches_generator_classification_train = []
            loss_batches_generator_reconstruction_train = []
            loss_batches_generator_eval = []
            loss_batches_generator_classification_eval = []
            loss_batches_generator_reconstruction_eval = []

            # Train mode setting
            self._discriminator.train()
            self._generator.train()

            # Iteration across batches (train)
            for databatch_train in self._loader_train:
                # Data extraction
                features_noise_perturbed_train = databatch_train['features_noise_perturbed']
                missing_mask_train = databatch_train['missing_mask']
                hint_train = databatch_train['hint']

                # Discriminator updating
                # feature generation
                _, features_combined_train = self._generator.forward(
                    features_noise=features_noise_perturbed_train, missing_mask=missing_mask_train)
                # feature discrimination
                real_imputed_probabilities_train = self._discriminator.forward(
                    features_combined=features_combined_train, hint=hint_train)
                # loss function calculation
                loss_discriminator_train = self._lossfun_discriminator(
                    real_imputed_probabilities=real_imputed_probabilities_train, missing_mask=missing_mask_train)
                # parameter updating
                self._optimizer_discriminator.zero_grad()
                loss_discriminator_train.backward()
                self._optimizer_discriminator.step()

                # Generator updating
                # feature generation
                features_generated_train_, features_combined_train_ = self._generator.forward(
                    features_noise=features_noise_perturbed_train, missing_mask=missing_mask_train)
                # feature discrimination
                real_imputed_probabilities_train_ = self._discriminator.forward(
                    features_combined=features_combined_train_, hint=hint_train)
                # loss function calculation
                loss_generator_train, loss_gen_clas_train, loss_gen_recons_train = self._lossfun_generator(
                    features_noise_perturbed=features_noise_perturbed_train, missing_mask=missing_mask_train,
                    features_generated=features_generated_train_,
                    real_imputed_probabilities=real_imputed_probabilities_train_)
                # parameter updating
                self._optimizer_generator.zero_grad()
                loss_generator_train.backward()
                self._optimizer_generator.step()

                # Losses storage
                loss_batches_discriminator_train.append(loss_discriminator_train.item())
                loss_batches_generator_train.append(loss_generator_train.item())
                loss_batches_generator_classification_train.append(loss_gen_clas_train.item())
                loss_batches_generator_reconstruction_train.append(loss_gen_recons_train.item())

            # Evaluation
            with no_grad():
                # Evaluation mode setting
                self._discriminator.eval()
                self._generator.eval()

                # Iteration across batches (evaluation)
                for databatch_eval in self._loader_eval:
                    # Data extraction
                    features_noise_perturbed_eval = databatch_eval['features_noise_perturbed']
                    missing_mask_eval = databatch_eval['missing_mask']
                    hint_eval = databatch_eval['hint']

                    # Feature generation
                    features_generated_eval, features_combined_eval = self._generator.forward(
                        features_noise=features_noise_perturbed_eval, missing_mask=missing_mask_eval)

                    # Feature discrimination
                    real_imputed_probabilities_eval = self._discriminator.forward(
                        features_combined=features_combined_eval, hint=hint_eval)

                    # Losses calculation
                    # discriminator
                    loss_discriminator_eval = self._lossfun_discriminator(
                        real_imputed_probabilities=real_imputed_probabilities_eval, missing_mask=missing_mask_eval)
                    # generator
                    loss_generator_eval, loss_gen_clas_eval, loss_gen_recons_eval = self._lossfun_generator(
                        features_noise_perturbed=features_noise_perturbed_eval, missing_mask=missing_mask_eval,
                        features_generated=features_generated_eval,
                        real_imputed_probabilities=real_imputed_probabilities_eval)

                    # Losses storage
                    loss_batches_discriminator_eval.append(loss_discriminator_eval.item())
                    loss_batches_generator_eval.append(loss_generator_eval.item())
                    loss_batches_generator_classification_eval.append(loss_gen_clas_eval.item())
                    loss_batches_generator_reconstruction_eval.append(loss_gen_recons_eval.item())

            # Loss averaging and storage
            # discriminator
            self._loss_epochs_discriminator_train.append(self._calculate_mean_loss(loss_batches_discriminator_train))
            self._loss_epochs_discriminator_eval.append(self._calculate_mean_loss(loss_batches_discriminator_eval))
            # generator
            self._loss_epochs_generator_train.append(self._calculate_mean_loss(loss_batches_generator_train))
            self._loss_epochs_generator_classification_train.append(
                self._calculate_mean_loss(loss_batches_generator_classification_train))
            self._loss_epochs_generator_reconstruction_train.append(
                self._calculate_mean_loss(loss_batches_generator_reconstruction_train))
            self._loss_epochs_generator_eval.append(self._calculate_mean_loss(loss_batches_generator_eval))
            self._loss_epochs_generator_classification_eval.append(
                self._calculate_mean_loss(loss_batches_generator_classification_eval))
            self._loss_epochs_generator_reconstruction_eval.append(
                self._calculate_mean_loss(loss_batches_generator_reconstruction_eval))

    # LOSS AVERAGING
    @staticmethod
    def _calculate_mean_loss(loss_values: list) -> float:
        # Calculation
        loss_mean = sum(loss_values) / len(loss_values)

        # Output
        return loss_mean

    # LOSSES DISPLAY
    def _display_losses(self, epoch: int):
        epoch_str = str(epoch)

        loss_disc_train = str(round(self._loss_epochs_discriminator_train[epoch], self._number_decimals2display))
        loss_gen_train = str(round(self._loss_epochs_generator_train[epoch], self._number_decimals2display))
        loss_disc_eval = str(round(self._loss_epochs_discriminator_eval[epoch], self._number_decimals2display))
        loss_gen_eval = str(round(self._loss_epochs_generator_eval[epoch], self._number_decimals2display))

        message_train = ['Eph:', epoch_str, 'DisTr:', loss_disc_train, 'GenTr:', loss_gen_train]
        message_eval = ['DisEv:', loss_disc_eval, 'GenEv:', loss_gen_eval]

        print(' '.join(message_train + message_eval))

    # LOSSES PLOTTING
    def plot_losses(self):
        epochs = [e for e in range(self._number_epochs)]

        mpl.figure()

        mpl.plot(epochs, self._loss_epochs_discriminator_eval, marker=self._marker, linewidth=self._linewidth,
                 markersize=self._markersize, markeredgecolor=self._markeredgecolor,
                 markeredgewidth=self._markeredgewidth, label='Discriminator')
        mpl.plot(epochs, self._loss_epochs_generator_eval, marker=self._marker, linewidth=self._linewidth,
                 markersize=self._markersize, markeredgecolor=self._markeredgecolor,
                 markeredgewidth=self._markeredgewidth, label='Generator')

        mpl.ylabel('Number epochs', fontsize=14)
        mpl.ylabel('Loss value', fontsize=14)

        mpl.title('Discriminator and generator losses validation', fontsize=16)
        mpl.xticks(rotation=45, fontsize=14)
        mpl.yticks(fontsize=14)

        mpl.legend(loc='lower left', fontsize=14)
        mpl.grid(color='grey', linestyle='dashed', linewidth=0.5)
        mpl.show()
