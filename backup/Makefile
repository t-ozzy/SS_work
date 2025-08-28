target = spot_diff.py
result = ./result.png
color = ./color_diff.py

all:
	python3 $(target)

color:
	python3 $(color) --debug

debug:
	python3 $(target) --debug

clean:
	$(RM) $(result)

fclean:
	$(RM) ./*.png

re: fclean all