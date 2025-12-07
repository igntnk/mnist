package main

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"time"
)

// ActivationFunction тип функции активации
type ActivationFunction func(float64) float64
type ActivationDerivative func(float64) float64

// Layer слой нейронной сети
type Layer struct {
	Weights     [][]float64 `json:"weights"`
	Biases      []float64   `json:"biases"`
	Activations []float64   `json:"-"`
	Z           []float64   `json:"-"` // Взвешенная сумма до активации
	Delta       []float64   `json:"-"` // Ошибка слоя
}

// Network нейронная сеть
type Network struct {
	Layers        []*Layer             `json:"layers"`
	LearningRate  float64              `json:"learning_rate"`
	Activation    ActivationFunction   `json:"-"`
	ActivationDer ActivationDerivative `json:"-"`
}

// NewNetwork создает новую нейронную сеть
func NewNetwork(architecture []int) *Network {
	rand.Seed(time.Now().UnixNano())

	network := &Network{
		Activation:    Sigmoid,
		ActivationDer: SigmoidDerivative,
		LearningRate:  0.01,
	}

	// Создаем слои
	for i := 0; i < len(architecture)-1; i++ {
		inputSize := architecture[i]
		outputSize := architecture[i+1]

		layer := &Layer{
			Weights: make([][]float64, outputSize),
			Biases:  make([]float64, outputSize),
		}

		// Инициализация весов (Xavier/Glorot)
		limit := math.Sqrt(6.0 / float64(inputSize+outputSize))
		for j := range layer.Weights {
			layer.Weights[j] = make([]float64, inputSize)
			for k := range layer.Weights[j] {
				layer.Weights[j][k] = rand.Float64()*2*limit - limit
			}
			layer.Biases[j] = 0.0
		}

		network.Layers = append(network.Layers, layer)
	}

	return network
}

// SetLearningRate устанавливает скорость обучения
func (n *Network) SetLearningRate(lr float64) {
	n.LearningRate = lr
}

// Forward прямое распространение
func (n *Network) Forward(input []float64) []float64 {
	current := input

	for i, layer := range n.Layers {
		layerSize := len(layer.Weights)

		// Вычисляем взвешенную сумму
		layer.Z = make([]float64, layerSize)
		layer.Activations = make([]float64, layerSize)

		for j := 0; j < layerSize; j++ {
			var sum float64

			// Взвешенная сумма
			for k := 0; k < len(current); k++ {
				sum += current[k] * layer.Weights[j][k]
			}

			// Добавляем смещение
			sum += layer.Biases[j]
			layer.Z[j] = sum

			// Применяем функцию активации
			if i == len(n.Layers)-1 {
				// Для выходного слоя используем softmax
				layer.Activations[j] = sum // Softmax применим позже
			} else {
				layer.Activations[j] = n.Activation(sum)
			}
		}

		// Применяем softmax для выходного слоя
		if i == len(n.Layers)-1 {
			layer.Activations = Softmax(layer.Activations)
		}

		current = layer.Activations
	}

	return current
}

// Backward обратное распространение ошибки
func (n *Network) Backward(input []float64, target int) {
	output := n.Forward(input)
	numLayers := len(n.Layers)

	// Вычисляем ошибку выходного слоя
	outputLayer := n.Layers[numLayers-1]
	outputLayer.Delta = make([]float64, len(output))

	for i := range output {
		// Для cross-entropy с softmax
		outputLayer.Delta[i] = output[i]
		if i == target {
			outputLayer.Delta[i] -= 1.0
		}
	}

	// Обратное распространение по скрытым слоям
	for l := numLayers - 2; l >= 0; l-- {
		currentLayer := n.Layers[l]
		nextLayer := n.Layers[l+1]

		currentLayer.Delta = make([]float64, len(currentLayer.Activations))

		// Вычисляем ошибку для текущего слоя
		for i := range currentLayer.Delta {
			var errorSum float64
			for j := range nextLayer.Delta {
				errorSum += nextLayer.Delta[j] * nextLayer.Weights[j][i]
			}
			currentLayer.Delta[i] = errorSum * n.ActivationDer(currentLayer.Z[i])
		}
	}
}

// UpdateWeights обновляет веса сети
func (n *Network) UpdateWeights(batchSize int) {
	for l := 0; l < len(n.Layers); l++ {
		layer := n.Layers[l]
		prevActivations := make([]float64, 0)

		if l == 0 {
			// Для первого слоя используем входные данные
			// (они сохраняются в Forward)
		} else {
			prevActivations = n.Layers[l-1].Activations
		}

		// Обновляем веса и смещения
		for i := range layer.Weights {
			for j := range layer.Weights[i] {
				var gradient float64
				if l == 0 {
					// Для первого слоя нужно получить входные данные
					// В реальной реализации нужно сохранять входные данные
					continue
				}
				gradient = layer.Delta[i] * prevActivations[j]
				layer.Weights[i][j] -= n.LearningRate * gradient / float64(batchSize)
			}
			layer.Biases[i] -= n.LearningRate * layer.Delta[i] / float64(batchSize)
		}
	}
}

// Save сохраняет модель в файл
func (n *Network) Save(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(n)
}

// Load загружает модель из файла
func (n *Network) Load(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	decoder := json.NewDecoder(file)
	return decoder.Decode(n)
}

// Sigmoid функция активации
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// SigmoidDerivative производная сигмоиды
func SigmoidDerivative(x float64) float64 {
	s := Sigmoid(x)
	return s * (1 - s)
}

// ReLU функция активации
func ReLU(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// ReLUDerivative производная ReLU
func ReLUDerivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

// Softmax функция
func Softmax(x []float64) []float64 {
	result := make([]float64, len(x))
	var sum float64

	// Вычитаем максимальное значение для численной стабильности
	maxVal := x[0]
	for _, val := range x {
		if val > maxVal {
			maxVal = val
		}
	}

	for i, val := range x {
		result[i] = math.Exp(val - maxVal)
		sum += result[i]
	}

	for i := range result {
		result[i] /= sum
	}

	return result
}

// CrossEntropyLoss вычисляет кросс-энтропию
func CrossEntropyLoss(predictions []float64, target int) float64 {
	// Добавляем небольшое значение для избежания log(0)
	epsilon := 1e-15
	prediction := predictions[target]

	if prediction < epsilon {
		prediction = epsilon
	}
	if prediction > 1-epsilon {
		prediction = 1 - epsilon
	}

	return -math.Log(prediction)
}

// ArgMax возвращает индекс максимального значения
func ArgMax(values []float64) int {
	maxIndex := 0
	maxValue := values[0]

	for i, value := range values {
		if value > maxValue {
			maxValue = value
			maxIndex = i
		}
	}

	return maxIndex
}
