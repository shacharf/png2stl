# png2stl
Convert png as height map to stl

```bash
 png2stl.py --image  im.png --size 40 40 --height 4 --outstl model.stl
```

# Changlog
* 22-02-21 - Added support for non-rectangular shapes
  	     Have a "bug" of the size of the shape
	     
# TODO
* Fix bug - does not seem to generate watertight geometry
* flip normals
* Documentation
* Add tests
