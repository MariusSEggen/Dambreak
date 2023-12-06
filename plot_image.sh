INPUT=$1
OUTPUT=`echo $1 | sed s/data/images/ | sed s/\.txt/.png/`
cat <<EOPLOT | gnuplot -
set style data dots
set term png
set xrange[-0.25:3.75]
set yrange[-0.1:1.0]
set output "${OUTPUT}"
plot "${INPUT}" using 2:3
EOPLOT
