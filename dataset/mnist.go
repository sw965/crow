package dataset

const (
	defaultBaseURL = "https://github.com/sw965/crow/releases/download/v0.2.0-test/"

	// MNIST
	mnistTrainInputsFileName = "mnist_train_flat_binary_imgs.gob"
	mnistTrainLabelsFileName = "mnist_train_int_labels.gob"
	mnistTestInputsFileName  = "mnist_test_flat_binary_imgs.gob"
	mnistTestLabelsFileName  = "mnist_test_int_labels.gob"

	// Fashion-MNIST
	fashionTrainInputsFileName = "fashion_mnist_train_flat_binary_imgs.gob"
	fashionTrainLabelsFileName = "fashion_mnist_train_int_labels.gob"
	fashionTestInputsFileName  = "fashion_mnist_test_flat_binary_imgs.gob"
	fashionTestLabelsFileName  = "fashion_mnist_test_int_labels.gob"
)

func LoadMNIST(logf LogFunc) (Binary[int], error) {
	dataDir, err := getCacheDir()
	if err != nil {
		return Binary[int]{}, err
	}
	return loadBinary[int](defaultBaseURL, dataDir, mnistTrainInputsFileName, mnistTrainLabelsFileName, mnistTestInputsFileName, mnistTestLabelsFileName, logf)
}

func LoadFashionMNIST(logf LogFunc) (Binary[int], error) {
	dataDir, err := getCacheDir()
	if err != nil {
		return Binary[int]{}, err
	}
	return loadBinary[int](defaultBaseURL, dataDir, fashionTrainInputsFileName, fashionTrainLabelsFileName, fashionTestInputsFileName, fashionTestLabelsFileName, logf)
}
