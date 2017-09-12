#Learning dictionaries. it is a key value pair and keys are immutable, meaning if we try to add new item with the same key it will just replcae it.
fruit = {"orange": "a sweet orange color fruit",
         "apple": "good for making cider",
         "lemon": "a sour, yellow citrus fruit",
         "grape": "a small fruit, growing in bunches",
         "lime": " a yellow colored fruit"}
#.keys and .values returns the keys and values and they are called views
print(fruit.keys())
print(fruit.values())
# dictionary will not be in sorted order and it will change every time, but you can do that using lists or sorted method
fruit_list=sorted(list(fruit))
for i in fruit_list:
    print("{} is a {}". format(i, fruit[i]))
# you can also directly use without creating lists
for i in sorted(fruit.keys()):
    print("{} is a {}".format(i, fruit[i]))
# dictionary.items() returns a tuple like object and we can use that to create tuples
f_tuple= tuple(fruit.items())
for i in f_tuple:
    fruit , desc = i
    print("{} is a {}".format(fruit,desc))
# we can use dict function to create a dictionary again
fruit_dict = dict(f_tuple)
print(fruit_dict)