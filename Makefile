CFLAGS+= -fopenmp -O2
LDLIBS+= -lm
TARGETS=wcsph_dambreak
IMAGES=$(shell find data/*.txt | sed s/data/images/g | sed s/\.txt/.png/g)
.PHONY: all dirs plot clean
all: dirs ${TARGETS}
wcsph_dambreak: wcsph_dambreak.c wcsph_dambreak.h
dirs:
	mkdir -p data images
plot: ${IMAGES}
images/%.png: data/%.txt
	./plot_image.sh $<
clean:
	-rm -f data/*.txt images/*.png ${TARGETS}
