package main

import (
	"fmt"
	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/driver/desktop"
	"fyne.io/fyne/v2/widget"
	"log"
	"math/rand"
	"time"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("=== Нейронная сеть для распознавания рукописных цифр MNIST ===")

	// 1. Загрузка данных MNIST
	fmt.Println("\n1. Загрузка данных MNIST...")
	trainImages, trainLabels, testImages, testLabels, err := LoadMNISTFromBin()
	if err != nil {
		log.Fatal("Ошибка загрузки данных:", err)
	}

	fmt.Printf("Загружено %d обучающих и %d тестовых изображений\n",
		len(trainImages), len(testImages))

	// 2. Создание нейронной сети
	fmt.Println("\n2. Создание нейронной сети...")
	network := NewNetwork([]int{784, 128, 64, 10}) // 784 входа, 2 скрытых слоя, 10 выходов
	network.SetLearningRate(0.1)

	// 3. Обучение сети
	fmt.Println("\n3. Начало обучения...")
	epochs := 150
	batchSize := 32

	trainLosses := make([]float64, epochs)
	trainAccuracies := make([]float64, epochs)

	for epoch := 0; epoch < epochs; epoch++ {
		startTime := time.Now()

		// Перемешиваем данные
		shuffledIndices := rand.Perm(len(trainImages))

		var epochLoss float64
		var correct int

		// Обучение мини-батчами
		for i := 0; i < len(shuffledIndices); i += batchSize {
			end := i + batchSize
			if end > len(shuffledIndices) {
				end = len(shuffledIndices)
			}

			batchIndices := shuffledIndices[i:end]
			batchLoss := 0.0
			batchCorrect := 0

			// Прямое распространение и обратное распространение для батча
			for _, idx := range batchIndices {
				image := trainImages[idx]
				label := trainLabels[idx]

				// Прямое распространение
				output := network.Forward(image)

				// Вычисляем потерю и точность
				batchLoss += CrossEntropyLoss(output, label)
				if ArgMax(output) == label {
					batchCorrect++
				}

				// Обратное распространение ошибки
				network.Backward(image, label)
			}

			// Обновление весов после батча
			network.UpdateWeights(len(batchIndices))

			epochLoss += batchLoss
			correct += batchCorrect
		}

		// Статистика эпохи
		avgLoss := epochLoss / float64(len(trainImages))
		accuracy := float64(correct) / float64(len(trainImages))

		trainLosses[epoch] = avgLoss
		trainAccuracies[epoch] = accuracy

		elapsed := time.Since(startTime)

		fmt.Printf("Эпоха %d/%d | Loss: %.4f | Accuracy: %.2f%% | Время: %v\n",
			epoch+1, epochs, avgLoss, accuracy*100, elapsed)

		// Тестирование после каждой эпохи
		if (epoch+1)%2 == 0 {
			testAccuracy := Evaluate(network, testImages, testLabels)
			fmt.Printf("  Тестовая точность: %.2f%%\n", testAccuracy*100)
		}
	}

	// 4. Финальное тестирование
	fmt.Println("\n4. Финальное тестирование...")
	testAccuracy := Evaluate(network, testImages, testLabels)
	fmt.Printf("Финальная точность на тестовой выборке: %.2f%%\n", testAccuracy*100)

	// 5. Визуализация результатов
	fmt.Println("\n5. Создание графиков...")
	if err := PlotTrainingResults(trainLosses, trainAccuracies); err != nil {
		fmt.Printf("Ошибка создания графиков: %v\n", err)
	} else {
		fmt.Println("Графики сохранены в training_results.png")
	}

	// 6. Демонстрация предсказаний
	fmt.Println("\n6. Демонстрация предсказаний...")
	ShowPredictions(network, testImages, testLabels, 10)

	// 7. Сохранение модели
	fmt.Println("\n7. Сохранение модели...")
	if err := network.Save("mnist_model.json"); err != nil {
		fmt.Printf("Ошибка сохранения модели: %v\n", err)
	} else {
		fmt.Println("Модель сохранена в mnist_model.json")
	}

	a := app.New()
	w := a.NewWindow("Number Predictor")

	grid := NewDrawGrid()
	clearBtn := widget.NewButton("Очистить", func() {
		grid.Clear()
	})

	label := widget.NewLabel("Тут будет отображаться предсказание сети")

	loadToNetworkBtn := widget.NewButton("Получить предсказание", func() {
		input := grid.getDataForPredict()
		output := network.Forward(input)
		prediction := ArgMax(output)
		confidence := output[prediction]

		label.SetText(fmt.Sprintf("Нейронная сеть думает, что это цифра - %d \n Она уверрена в этом на %.2f%%", prediction, confidence*100))
	})
	w.SetContent(container.NewHBox(container.NewVBox(
		grid,
		clearBtn,
	),
		loadToNetworkBtn,
		label,
	))
	w.Resize(fyne.NewSize(GridSize*PixelSize+10, GridSize*PixelSize+10))

	if desk, ok := a.Driver().(desktop.Driver); ok {
		_ = desk // desktop events if needed
	}

	w.ShowAndRun()
}
