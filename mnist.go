// Используем готовый датасет через Python и сохраняем в бинарном формате
package main

import "os"

// LoadMNISTFromBin загружает данные из бинарных файлов
func LoadMNISTFromBin() ([][]float64, []int, [][]float64, []int, error) {
	// Чтение тренировочных изображений
	trainImagesData, err := os.ReadFile("data/train-images.bin")
	if err != nil {
		return nil, nil, nil, nil, err
	}

	trainLabelsData, err := os.ReadFile("data/train-labels.bin")
	if err != nil {
		return nil, nil, nil, nil, err
	}

	// Чтение тестовых изображений
	testImagesData, err := os.ReadFile("data/test-images.bin")
	if err != nil {
		return nil, nil, nil, nil, err
	}

	testLabelsData, err := os.ReadFile("data/test-labels.bin")
	if err != nil {
		return nil, nil, nil, nil, err
	}

	// Конвертация тренировочных данных
	numTrain := 60000
	trainImages := make([][]float64, numTrain)
	trainLabels := make([]int, numTrain)

	for i := 0; i < numTrain; i++ {
		trainImages[i] = make([]float64, 784)
		for j := 0; j < 784; j++ {
			trainImages[i][j] = float64(trainImagesData[i*784+j]) / 255.0
		}
		trainLabels[i] = int(trainLabelsData[i])
	}

	// Конвертация тестовых данных
	numTest := 10000
	testImages := make([][]float64, numTest)
	testLabels := make([]int, numTest)

	for i := 0; i < numTest; i++ {
		testImages[i] = make([]float64, 784)
		for j := 0; j < 784; j++ {
			testImages[i][j] = float64(testImagesData[i*784+j]) / 255.0
		}
		testLabels[i] = int(testLabelsData[i])
	}

	return trainImages, trainLabels, testImages, testLabels, nil
}
