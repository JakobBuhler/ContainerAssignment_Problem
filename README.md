    Description
    -----------
    The Container Assignment Problem involves assigning freight containers to transportation modes to minimize
    costs while meeting delivery deadlines. This problem is critical in logistics for optimizing the use of multimodal
    transport networks, which include various transportation modes like trucks and barges. Efficient container
    assignment can lead to significant cost savings and improved service levels, especially in dynamic environments
    where real-time decision-making is essential.
    The Container Assignment Problem tries to find the best mode of transportation for each container, in this case
    the truck or barge mode while recognizing capacity constraints for each transportation route.
    The Problem creates a QUBO-Matrix with the size
    N+M*K x N+M*K
    with N containers, M tracks of transportation, and K amount of Slack Variables for each track. This produces a
    solution vector where my decision variables represent if the container was transported by truck x = 1 or by
    barge mode x = 0.
