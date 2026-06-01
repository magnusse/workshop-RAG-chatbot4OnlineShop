from ragshop.sales_consultance.domain.model.intent import Intent


def test_parse_known_values():
    assert Intent.parse("PRODUCT_INFO") == Intent.PRODUCT_INFO
    assert Intent.parse(" offer_request ") == Intent.OFFER_REQUEST


def test_parse_unknown_falls_back_to_smalltalk():
    assert Intent.parse("garbage") == Intent.SMALLTALK
    assert Intent.parse("") == Intent.SMALLTALK
