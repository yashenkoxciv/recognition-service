# API

1. returns empty document {} if there are no matching subcategories
2. returns document {subcategory: <subcat_id>} where there is matching subcategory but no corresponding category
3. returns document {subcategory: <subcat_id>, category: <cat_id>} where there is matching subcategory and corresponding category

