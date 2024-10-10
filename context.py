
from word2number import w2n

def extract_order_items(tagged_tokens):
    order_items = {}
    quantity = None
    item = None

    for word, tag in tagged_tokens:
        # Look for numbers (CD = cardinal number in POS tagging)
        if tag == 'CD':
            quantity = int(word)  # Convert quantity to integer
        # Look for nouns (NN = noun, singular; NNS = noun, plural)
        elif tag in ['NN', 'NNS']:
            item = word.lower()  # Normalize to lowercase
            if quantity:
                order_items[item] = quantity  # Map item to quantity
                quantity = None  # Reset quantity after it's assigned

    return order_items


def validate_ordered_items(order_items,context,tag):
    menu = ["pizza", "lassi", "burger", "fries", "sandwich", "iced tea", "biryani", "raita", "rava dosa", "chole bhature", "samosa"]
    for item, quant in order_items.items():
        if isinstance(quant,str):
            quantity = w2n.word_to_num(quant)
        elif isinstance(quant,(int, float)):
            quantity = quant
        if tag=="remove order":
            quantity = -quantity
        if item in menu:
            if item not in context.keys() and quantity > 0:
                context[item] = quantity
            else:
                context[item] += quantity
    return context

