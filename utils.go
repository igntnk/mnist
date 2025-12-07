package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"os"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// Evaluate оценивает точность сети
func Evaluate(network *Network, images [][]float64, labels []int) float64 {
	correct := 0

	for i := 0; i < len(images); i++ {
		output := network.Forward(images[i])
		prediction := ArgMax(output)

		if prediction == labels[i] {
			correct++
		}
	}

	return float64(correct) / float64(len(images))
}

// PlotTrainingResults создает график обучения
func PlotTrainingResults(losses, accuracies []float64) error {
	p := plot.New()
	p.Title.Text = "График обучения"
	p.X.Label.Text = "Эпоха"
	p.Y.Label.Text = "Значение"

	// Создаем точки для loss
	lossPoints := make(plotter.XYs, len(losses))
	for i, loss := range losses {
		lossPoints[i].X = float64(i + 1)
		lossPoints[i].Y = loss
	}

	lossLine, err := plotter.NewLine(lossPoints)
	if err != nil {
		return err
	}
	lossLine.Color = color.RGBA{R: 255, G: 0, B: 0, A: 255}

	// Создаем точки для accuracy
	accPoints := make(plotter.XYs, len(accuracies))
	for i, acc := range accuracies {
		accPoints[i].X = float64(i + 1)
		accPoints[i].Y = acc * 100 // В процентах
	}

	accLine, err := plotter.NewLine(accPoints)
	if err != nil {
		return err
	}
	accLine.Color = color.RGBA{R: 0, G: 0, B: 255, A: 255}

	p.Add(lossLine, accLine)
	p.Legend.Add("Loss", lossLine)
	p.Legend.Add("Accuracy (%)", accLine)

	// Сохраняем график
	if err := p.Save(10*vg.Inch, 6*vg.Inch, "training_results.png"); err != nil {
		return err
	}

	return nil
}

// ShowPredictions показывает примеры предсказаний
func ShowPredictions(network *Network, images [][]float64, labels []int, numExamples int) {
	fmt.Println("\nПримеры предсказаний:")
	fmt.Println("=====================")

	for i := 0; i < numExamples && i < len(images); i++ {
		output := network.Forward(images[i])
		prediction := ArgMax(output)
		confidence := output[prediction]

		fmt.Printf("Изображение %d:\n", i+1)
		fmt.Printf("  Реальная цифра: %d\n", labels[i])
		fmt.Printf("  Предсказание:   %d (уверенность: %.2f%%)\n",
			prediction, confidence*100)

		if prediction == labels[i] {
			fmt.Printf("  Результат: ✓ Правильно\n")
		} else {
			fmt.Printf("  Результат: ✗ Ошибка\n")
		}
		fmt.Println()
	}
}

// SaveImageAsPNG сохраняет изображение MNIST как PNG
func SaveImageAsPNG(imageData []float64, filename string, label int) error {
	img := image.NewGray(image.Rect(0, 0, 28, 28))

	for y := 0; y < 28; y++ {
		for x := 0; x < 28; x++ {
			idx := y*28 + x
			grayValue := uint8(imageData[idx] * 255)
			img.SetGray(x, y, color.Gray{Y: grayValue})
		}
	}

	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	return png.Encode(file, img)
}

// CreateSampleImages создает примеры изображений
func CreateSampleImages(images [][]float64, labels []int, num int) {
	os.MkdirAll("samples", 0755)

	for i := 0; i < num && i < len(images); i++ {
		filename := fmt.Sprintf("samples/sample_%d_label_%d.png", i, labels[i])
		if err := SaveImageAsPNG(images[i], filename, labels[i]); err != nil {
			log.Printf("Ошибка сохранения изображения %d: %v", i, err)
		} else {
			fmt.Printf("Сохранено: %s\n", filename)
		}
	}
}
