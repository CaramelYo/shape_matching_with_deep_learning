get_image_contour: get_image_contour.o mat_image.o
	g++ get_image_contour.o mat_image.o -o get_image_contour -lstdc++fs `pkg-config --cflags --libs opencv`
	make clean
get_image_contour.o: get_image_contour.cpp
	g++ get_image_contour.cpp -c
mat_image.o: mat_image.cpp
	g++ mat_image.cpp -c
clean:
	rm -rf get_image_contour.o mat_image.o