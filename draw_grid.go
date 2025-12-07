package main

import (
	"image/color"
	"math"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/driver/desktop"
	"fyne.io/fyne/v2/widget"
)

const (
	GridSize  = 28
	PixelSize = 20
)

type DrawGrid struct {
	widget.BaseWidget
	Data      [][]float64
	mouseDown bool
}

func NewDrawGrid() *DrawGrid {
	d := &DrawGrid{
		Data: make([][]float64, GridSize),
	}
	for i := range d.Data {
		d.Data[i] = make([]float64, GridSize)
	}
	d.ExtendBaseWidget(d)
	return d
}

func (d *DrawGrid) Clear() {
	for y := 0; y < GridSize; y++ {
		for x := 0; x < GridSize; x++ {
			d.Data[y][x] = 0
		}
	}
	d.Refresh()
}

func (d *DrawGrid) CreateRenderer() fyne.WidgetRenderer {
	rects := make([]fyne.CanvasObject, 0, GridSize*GridSize)
	for y := 0; y < GridSize; y++ {
		for x := 0; x < GridSize; x++ {
			r := canvas.NewRectangle(color.White)
			r.StrokeColor = color.RGBA{200, 200, 200, 255}
			r.StrokeWidth = 1
			rects = append(rects, r)
		}
	}
	return &drawGridRenderer{grid: d, rects: rects}
}

type drawGridRenderer struct {
	grid  *DrawGrid
	rects []fyne.CanvasObject
}

func (r *drawGridRenderer) Layout(size fyne.Size) {
	for y := 0; y < GridSize; y++ {
		for x := 0; x < GridSize; x++ {
			idx := y*GridSize + x
			r.rects[idx].Resize(fyne.NewSize(PixelSize, PixelSize))
			r.rects[idx].Move(fyne.NewPos(float32(x*PixelSize), float32(y*PixelSize)))
		}
	}
}

func (r *drawGridRenderer) MinSize() fyne.Size {
	return fyne.NewSize(GridSize*PixelSize, GridSize*PixelSize)
}

func (r *drawGridRenderer) Refresh() {
	for y := 0; y < GridSize; y++ {
		for x := 0; x < GridSize; x++ {
			idx := y*GridSize + x
			gray := uint8(255 - r.grid.Data[y][x]*255)
			r.rects[idx].(*canvas.Rectangle).FillColor = color.RGBA{gray, gray, gray, 255}
			r.rects[idx].Refresh()
		}
	}
}

func (r *drawGridRenderer) Objects() []fyne.CanvasObject { return r.rects }
func (r *drawGridRenderer) Destroy()                     {}

func (d *DrawGrid) mouseDraw(pos fyne.Position) {
	cx := float64(pos.X)
	cy := float64(pos.Y)

	for y := 0; y < GridSize; y++ {
		for x := 0; x < GridSize; x++ {
			px := float64(x*PixelSize + PixelSize/2)
			py := float64(y*PixelSize + PixelSize/2)
			dx := cx - px
			dy := cy - py
			dist := math.Sqrt(dx*dx + dy*dy)
			if dist < PixelSize*1.5 { // радиус влияния
				// чем ближе к центру пикселя — тем сильнее цвет
				influence := 1 - dist/(PixelSize*1.5)
				if influence < 0 {
					influence = 0
				}
				// аккумулируем, но не больше 1
				d.Data[y][x] += influence
				if d.Data[y][x] > 1 {
					d.Data[y][x] = 1
				}
			}
		}
	}

	d.Refresh()
}

func (d *DrawGrid) getDataForPredict() []float64 {
	result := make([]float64, GridSize*GridSize)
	index := 0
	for y := 0; y < GridSize; y++ {
		for x := 0; x < GridSize; x++ {
			result[index] = d.Data[y][x]
			index++
		}
	}
	return result
}

func (d *DrawGrid) MouseDown(ev *desktop.MouseEvent) {
	d.mouseDown = true
	d.mouseDraw(ev.Position)
}

func (d *DrawGrid) MouseUp(ev *desktop.MouseEvent) {
	d.mouseDown = false
}

func (d *DrawGrid) MouseMoved(ev *desktop.MouseEvent) {
	if d.mouseDown {
		go func() {
			fyne.DoAndWait(func() {
				d.Refresh()
			})
		}()

		d.mouseDraw(ev.Position)
	}
}

func (d *DrawGrid) MouseIn(*desktop.MouseEvent) {}

func (d *DrawGrid) MouseOut() {}
