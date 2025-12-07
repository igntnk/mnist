package main

import (
	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/driver/desktop"
	"fyne.io/fyne/v2/widget"
	"image/color"
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

func (d *DrawGrid) CreateRenderer() fyne.WidgetRenderer {
	objects := make([]fyne.CanvasObject, 0, GridSize*GridSize)

	for y := 0; y < GridSize; y++ {
		for x := 0; x < GridSize; x++ {
			rect := canvas.NewRectangle(color.White)
			rect.StrokeColor = color.RGBA{200, 200, 200, 255}
			rect.StrokeWidth = 1
			objects = append(objects, rect)
		}
	}

	return &drawGridRenderer{grid: d, rects: objects}
}

type drawGridRenderer struct {
	grid  *DrawGrid
	rects []fyne.CanvasObject
}

func (r *drawGridRenderer) Layout(size fyne.Size) {
	for y := 0; y < GridSize; y++ {
		for x := 0; x < GridSize; x++ {
			idx := y*GridSize + x
			px := float32(x * PixelSize)
			py := float32(y * PixelSize)
			r.rects[idx].Resize(fyne.NewSize(PixelSize, PixelSize))
			r.rects[idx].Move(fyne.NewPos(px, py))
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
			rect := r.rects[idx].(*canvas.Rectangle)

			if r.grid.Data[y][x] == 1 {
				rect.FillColor = color.Black
			} else {
				rect.FillColor = color.White
			}

			rect.Refresh()
		}
	}
}

func (r *drawGridRenderer) Objects() []fyne.CanvasObject { return r.rects }
func (r *drawGridRenderer) Destroy()                     {}

func (d *DrawGrid) pointToPixel(p fyne.Position) (int, int, bool) {
	x := int(p.X) / PixelSize
	y := int(p.Y) / PixelSize
	if x < 0 || x >= GridSize || y < 0 || y >= GridSize {
		return 0, 0, false
	}
	return x, y, true
}

func (d *DrawGrid) mouseDraw(p fyne.Position) {
	x, y, ok := d.pointToPixel(p)
	if !ok {
		return
	}
	d.Data[y][x] = 1
	if y+1 < GridSize && x+1 < GridSize {
		d.Data[y+1][x+1] = 1
	}

	if y+1 < GridSize {
		d.Data[y+1][x] = 1
	}

	if x+1 < GridSize {
		d.Data[y][x+1] = 1
	}
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

func (d *DrawGrid) Clear() {
	for y := 0; y < GridSize; y++ {
		for x := 0; x < GridSize; x++ {
			d.Data[y][x] = 0
		}
	}
	d.Refresh()
}
