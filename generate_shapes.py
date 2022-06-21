import png
import random
import math
size = 64
img = []


def generate_square(img):
    squareSize = random.randint(5, size)
    minx = random.randint(0, size - squareSize)
    miny = random.randint(0, size - squareSize)
    (maxx, maxy) = (minx + squareSize, miny + squareSize)
    for y in range(size):
        row = []
        for x in range(size):
            if x >= minx and x <= maxx and y >= miny and y <= maxy:
                row.append(255)
            else:
                row.append(0)
        img.append(row)


def generate_circle(img):
    circleSize = random.randint(5, size * 0.5)
    circleSizePow2 = circleSize * circleSize
    circleX = random.randint(circleSize, size - circleSize)
    circleY = random.randint(circleSize, size - circleSize)
    for y in range(size):
        row = []
        for x in range(size):
            if (x - circleX) * (x - circleX) + (y - circleY) * (y - circleY) <= circleSizePow2:
                row.append(255)
            else:
                row.append(0)
        img.append(row)


def point_in_triangle(px, py, p0x, p0y, p1x, p1y, p2x, p2y):
    s = (p0x - p2x) * (py - p2y) - (p0y - p2y) * (px - p2x)
    t = (p1x - p0x) * (py - p0y) - (p1y - p0y) * (px - p0x)
    if (s < 0) != (t < 0) and s != 0 and t != 0:
        return False
    d = (p2x - p1x) * (py - p1y) - (p2y - p1y) * (px - p1x)
    return d == 0 or (d < 0) == (s + t <= 0)


def generate_triangle(img):
    triangleSize = random.randint(5, size)
    triangleHeight = (triangleSize * math.sqrt(3)) / 2
    triangleAx = random.randint(0, size - triangleSize)
    triangleAy = random.randint(int(triangleHeight), size)
    triangleBx = triangleAx + triangleSize
    triangleBy = triangleAy
    triangleCx = triangleAx + int(triangleSize / 2)
    triangleCy = triangleAy - triangleHeight
    for y in range(size):
        row = []
        for x in range(size):
            if point_in_triangle(x, y, triangleAx, triangleAy, triangleBx, triangleBy, triangleCx, triangleCy):
                row.append(255)
            else:
                row.append(0)

        img.append(row)


def generate_star(img):

    starWidth = random.randint(5, size)
    a = starWidth/2
    middleX = random.randint(math.floor(starWidth/2),
                             math.floor(size-starWidth/2))
    middleY = random.randint(math.floor(starWidth/2),
                             math.floor(size-starWidth/2))
    a32 = math.sqrt(3)*a/2
    Ax = middleX+a
    Ay = middleY
    Dx = middleX-a
    Dy = middleY
    Bx = middleX+a/2
    By = middleY+a32
    Cx = middleX-a/2
    Cy = middleY+a32
    Ex = middleX-a/2
    Ey = middleY-a32
    Fx = middleX+a/2
    Fy = middleY-a32
    row = []

    for y in range(size):
        row = []
        for x in range(size):
            color = 0
            if point_in_triangle(x, y, Dx, Dy, Bx, By, Fx, Fy):
                color = 255
            if point_in_triangle(x, y, Ax, Ay, Ex, Ey, Cx, Cy):
                color = 255
            row.append(color)
        img.append(row)


def main():
    samplesPerShape = 10000
    # generate squares
    for i in range(samplesPerShape):
        img = []
        generate_square(img)
        with open('data/square' + str(i) + '.png', 'wb') as f:
            w = png.Writer(size, size, greyscale=True)
            w.write(f, img)

   # generate circles
    for i in range(samplesPerShape):
        img = []
        generate_circle(img)
        with open('data/circle' + str(i) + '.png', 'wb') as f:
            w = png.Writer(size, size, greyscale=True)
            w.write(f, img)

    # generate triangles
    for i in range(samplesPerShape):
        img = []
        generate_triangle(img)
        with open('data/triangle' + str(i) + '.png', 'wb') as f:
            w = png.Writer(size, size, greyscale=True)
            w.write(f, img)
    # generate stars
    for i in range(samplesPerShape):
        img = []
        generate_star(img)
        with open('data/star' + str(i) + '.png', 'wb') as f:
            w = png.Writer(size, size, greyscale=True)
            w.write(f, img)


if __name__ == "__main__":
    main()
