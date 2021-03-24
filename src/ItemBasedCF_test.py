from . import ItemBasedCF

def test_ItemBasedCF():
    assert ItemBasedCF.apply("Jane") == "hello Jane"
