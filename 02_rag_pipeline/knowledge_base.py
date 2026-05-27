# Base de connaissances RAG enrichie
# Sources : Masyhuri (2022), Al-Ayed (2022), Nguyen & Chanut (2018),
#           De Caigny et al. (2020), Tirunillai & Tellis (2014), Archak et al. (2011)

knowledge_base = [

    # ── DIMENSION 1 : LIVRAISON ───────────────────────────────────────────────
    {
        "id": "liv_01", "dimension": "livraison", "source": "Masyhuri (2022)",
        "text": "[Delivery reliability] Fulfillment and reliability aim to fulfill customer orders correctly and deliver them on time. A negative outcome in fulfillment and reliability could affect overall customer satisfaction and ruin the future of e-commerce companies."
    },
    {
        "id": "liv_02", "dimension": "livraison", "source": "Masyhuri (2022)",
        "text": "[Delivery reliability] It is important for the company to provide service correctly by executing customer transactions correctly and delivering customer orders in a timely manner."
    },
    {
        "id": "liv_03", "dimension": "livraison", "source": "Masyhuri (2022)",
        "text": "[Delivery reliability] Order fulfillment is a key determinant of customer satisfaction. Fulfillment and reliability have a significant impact on e-customer satisfaction."
    },
    {
        "id": "liv_04", "dimension": "livraison", "source": "Nguyen & Chanut (2018)",
        "text": "[Delivery reliability] La fiabilité de la livraison et le traitement des commandes ont un impact significatif sur la qualité de service perçue. Les consonautes sont de plus en plus sensibles au service logistique."
    },
    {
        "id": "liv_05", "dimension": "livraison", "source": "Nguyen & Chanut (2018)",
        "text": "[Delivery speed] Les délais de livraison, la traçabilité en temps réel et les modalités de retour constituent des facteurs déterminants de la qualité perçue du service électronique et de l'intention de réachat."
    },
    {
        "id": "liv_06", "dimension": "livraison", "source": "Nguyen & Chanut (2018)",
        "text": "[Delivery speed] Questions clés pour le consonaute : la rapidité de la livraison, le respect du délai annoncé, la traçabilité des informations et les modalités de retour simplifiées."
    },
    {
        "id": "liv_07", "dimension": "livraison", "source": "Nguyen & Chanut (2018)",
        "text": "[Logistics service] Le service logistique est susceptible de créer un avantage concurrentiel pour les e-commerçants. La partie aval de l'achat reste un critère décisif d'évaluation."
    },
    {
        "id": "liv_08", "dimension": "livraison", "source": "Tirunillai & Tellis (2014)",
        "text": "[Delivery dimension] User-generated content allows firms to extract latent dimensions of customer satisfaction including objective dimensions such as delivery speed and reliability that dominate vertically differentiated markets."
    },
    {
        "id": "liv_09", "dimension": "livraison", "source": "Archak et al. (2011)",
        "text": "[Shipping feature] Consumer reviews frequently mention shipping and delivery as distinct product features with associated opinions, reflecting customer satisfaction or dissatisfaction with the fulfillment process."
    },
    {
        "id": "liv_10", "dimension": "livraison", "source": "Masyhuri (2022)",
        "text": "[Late delivery] Customers express dissatisfaction when orders arrive late, are damaged during shipping, or when tracking information is unavailable. Delivery failures directly reduce repurchase intentions."
    },
    {
        "id": "liv_11", "dimension": "livraison", "source": "Nguyen & Chanut (2018)",
        "text": "[On-time delivery] Customers are satisfied when products arrive before or on the announced delivery date. Surpassing delivery expectations positively impacts overall service quality perception."
    },

    # ── DIMENSION 2 : QUALITÉ PRODUIT ─────────────────────────────────────────
    {
        "id": "qua_01", "dimension": "qualite_produit", "source": "Masyhuri (2022)",
        "text": "[Product quality] Performance deterioration is a threat to customer loyalty. Even when a customer repurchases constantly, threats to loyalty arise from performance deterioration or product quality issues."
    },
    {
        "id": "qua_02", "dimension": "qualite_produit", "source": "Al-Ayed (2022)",
        "text": "[Product expectations] When the performance of a product is lower than expected, negative disconfirmation occurs, causing dissatisfaction and reducing repurchase intentions."
    },
    {
        "id": "qua_03", "dimension": "qualite_produit", "source": "Al-Ayed (2022)",
        "text": "[Product conformity] Positive or negative disconfirmation is formed when expectations before and after purchase are compared, which affects overall satisfaction with a product."
    },
    {
        "id": "qua_04", "dimension": "qualite_produit", "source": "De Caigny et al. (2020)",
        "text": "[Product features] Product reviews are multifaceted. The textual content of reviews is an important determinant of consumer choices, over and above the valence and volume of reviews, because products have multiple attributes."
    },
    {
        "id": "qua_05", "dimension": "qualite_produit", "source": "De Caigny et al. (2020)",
        "text": "[Product attributes] By compressing a complex review to a single number, we implicitly assume that product quality is one-dimensional, whereas products have multiple attributes and different attributes can have different levels of importance to consumers."
    },
    {
        "id": "qua_06", "dimension": "qualite_produit", "source": "De Caigny et al. (2020)",
        "text": "[Product quality dimension] Consumers evaluate product quality based on specific features such as picture quality, ease of use, battery life, or design. Negative evaluations of key product attributes directly reduce purchase likelihood."
    },
    {
        "id": "qua_07", "dimension": "qualite_produit", "source": "Tirunillai & Tellis (2014)",
        "text": "[Quality dimensions] Quality is a multidimensional construct. The dimensions of quality are critical because they constitute the basis on which consumers evaluate brands and firms compete with one another."
    },
    {
        "id": "qua_08", "dimension": "qualite_produit", "source": "Tirunillai & Tellis (2014)",
        "text": "[Objective quality] For vertically differentiated markets, objective dimensions dominate and are similar across markets. Customers explicitly mention product performance, durability, and conformity to description in reviews."
    },
    {
        "id": "qua_09", "dimension": "qualite_produit", "source": "Archak et al. (2011)",
        "text": "[Product feature opinions] Each opinion phrase about a product feature represents a dimension of consumer experience. Negative opinion phrases about key product features are associated with lower customer satisfaction and reduced sales."
    },
    {
        "id": "qua_10", "dimension": "qualite_produit", "source": "Nguyen & Chanut (2018)",
        "text": "[Product description] La qualité des contenus du site marchand, incluant l'exactitude et la complétude des informations produit, constitue une dimension récurrente de la qualité de service électronique."
    },
    {
        "id": "qua_11", "dimension": "qualite_produit", "source": "Masyhuri (2022)",
        "text": "[Damaged product] Customers express strong dissatisfaction when products arrive broken, damaged, or do not match the description. Product non-conformity is a primary driver of negative reviews and churn."
    },
    {
        "id": "qua_12", "dimension": "qualite_produit", "source": "De Caigny et al. (2020)",
        "text": "[Positive product quality] Positive product quality perceptions arise when products meet or exceed customer expectations. Customers mention excellent quality, great design, and ease of use as key drivers of satisfaction."
    },

    # ── DIMENSION 3 : SERVICE CLIENT ──────────────────────────────────────────
    {
        "id": "ser_01", "dimension": "service_client", "source": "Masyhuri (2022)",
        "text": "[Customer service] Customer service aims to answer and solve customers requests and problems as quickly as possible. The lack of pleasant customer experiences and personalized advice discourages online shopping."
    },
    {
        "id": "ser_02", "dimension": "service_client", "source": "Masyhuri (2022)",
        "text": "[Service responsiveness] A company's prompt response to customer inquiries has a positive effect on satisfaction. Poor customer service results in immediate dissatisfaction and customer loss."
    },
    {
        "id": "ser_03", "dimension": "service_client", "source": "Masyhuri (2022)",
        "text": "[Complaint handling] E-commerce firms should handle customers problems and complaints in a friendly and courteous manner. Customer service has a significant impact on online customer satisfaction."
    },
    {
        "id": "ser_04", "dimension": "service_client", "source": "Al-Ayed (2022)",
        "text": "[Care dimension] Care refers to the attention paid to the consumer before and after the act of purchase to maintain long-term relationships. It is a key determinant of e-customer loyalty."
    },
    {
        "id": "ser_05", "dimension": "service_client", "source": "Al-Ayed (2022)",
        "text": "[Contact interactivity] Contact interactivity is the degree to which a website facilitates interaction with customers. It is a determinant factor that influences the desire to return to the website."
    },
    {
        "id": "ser_06", "dimension": "service_client", "source": "Nguyen & Chanut (2018)",
        "text": "[Post-purchase service] La réactivité et le contact constituent des dimensions clés de la qualité de service électronique post-achat, notamment dans la gestion des réclamations et le traitement des retours."
    },
    {
        "id": "ser_07", "dimension": "service_client", "source": "Tirunillai & Tellis (2014)",
        "text": "[Service dimension] User-generated content captures customer satisfaction with service interactions. Customers explicitly mention support quality, response time, and problem resolution in reviews."
    },
    {
        "id": "ser_08", "dimension": "service_client", "source": "De Caigny et al. (2020)",
        "text": "[Service features] Consumer reviews mention service-related features such as customer support, return process, and after-sales assistance as distinct dimensions influencing overall satisfaction."
    },
    {
        "id": "ser_09", "dimension": "service_client", "source": "Masyhuri (2022)",
        "text": "[No response] Customers become highly dissatisfied when companies fail to respond to complaints or inquiries. Lack of customer service response is a strong signal of churn intent."
    },
    {
        "id": "ser_10", "dimension": "service_client", "source": "Archak et al. (2011)",
        "text": "[Support opinions] Opinion phrases about customer support and service interactions represent key dimensions in consumer reviews. Negative service opinions are strongly associated with product dissatisfaction."
    },

    # ── DIMENSION 4 : PRIX ────────────────────────────────────────────────────
    {
        "id": "pri_01", "dimension": "prix", "source": "Masyhuri (2022)",
        "text": "[Price motivation] More than 60% of customers worldwide cite the lowest price as their first reason for visiting an e-commerce website. However, quality of service is more important than low price for retention."
    },
    {
        "id": "pri_02", "dimension": "prix", "source": "Nguyen & Chanut (2018)",
        "text": "[Price vs service] Le prix ne présente pas toujours l'élément le plus important pour l'intention de revisite. La prestation de service est plus importante que le bas prix pour fidéliser le client."
    },
    {
        "id": "pri_03", "dimension": "prix", "source": "Al-Ayed (2022)",
        "text": "[Price competition] The Internet has created hyper-competitivity where transaction costs are low and customers have great flexibility to move from one website to another, creating competition based on price."
    },
    {
        "id": "pri_04", "dimension": "prix", "source": "Al-Ayed (2022)",
        "text": "[Switching cost] Switching barriers in e-commerce depend on the consumer's perception of money, time and effort required to change provider. Low switching costs make price a central factor in retention."
    },
    {
        "id": "pri_05", "dimension": "prix", "source": "De Caigny et al. (2020)",
        "text": "[Price-quality tradeoff] Consumers evaluate products based on multiple attributes including price-quality ratio. A product perceived as overpriced relative to its quality generates negative reviews and reduces repurchase intent."
    },
    {
        "id": "pri_06", "dimension": "prix", "source": "Tirunillai & Tellis (2014)",
        "text": "[Price dimension] Price and value for money constitute explicit dimensions mentioned in consumer reviews. Customers express dissatisfaction when perceived value does not match the price paid."
    },
    {
        "id": "pri_07", "dimension": "prix", "source": "Archak et al. (2011)",
        "text": "[Pricing power] Product features have different pricing power. Consumer reviews reveal which product attributes justify the price paid and which attributes generate dissatisfaction relative to price expectations."
    },
    {
        "id": "pri_08", "dimension": "prix", "source": "Masyhuri (2022)",
        "text": "[Value for money] Customers express dissatisfaction when they feel the product or service does not offer sufficient value for the price paid. Poor price-quality ratio is a recurring theme in negative reviews."
    },
]

print(f"Base de connaissances enrichie : {len(knowledge_base)} passages")
print(f"Dimensions : {set(p['dimension'] for p in knowledge_base)}")
for dim in ['livraison', 'qualite_produit', 'service_client', 'prix']:
    count = sum(1 for p in knowledge_base if p['dimension'] == dim)
    print(f"  {dim} : {count} passages")