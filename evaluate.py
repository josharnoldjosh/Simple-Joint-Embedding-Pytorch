import numpy
import torch
from scipy import spatial

def recall_score(text_to_image, image_to_text):
    """
    Prints the averaged Recall@K scores
    """
    (scores_1, text_to_image_scores) = list(zip(*text_to_image))
    (scores_2, image_to_text_scores) = list(zip(*image_to_text))

    # average recall scores
    text_to_image_scores = [sum(y) / len(y) for y in zip(*text_to_image_scores)]
    image_to_text_scores = [sum(y) / len(y) for y in zip(*image_to_text_scores)]

    (r1, r5, r10, medr) = text_to_image_scores
    print("[RECALL@K] Text to image scores: R@1: %.1f, R@5: %.1f, R@10: %.1f, Medr: %.1f" % (r1, r5, r10, medr))

    (r1, r5, r10, medr) = image_to_text_scores
    print("[RECALL@K] Image to text scores: R@1: %.1f, R@5: %.1f, R@10: %.1f, Medr: %.1f" % (r1, r5, r10, medr))    

    return sum(scores_1)+sum(scores_2)

def image_to_text(captions, images, npts=None):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts == None:
        npts = images.size()[0] / 5
        npts = int(npts)

    ranks = numpy.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[5 * index].unsqueeze(0)

        # Compute scores
        d = torch.mm(im, captions.t())
        d_sorted, inds = torch.sort(d, descending=True)
        inds = inds.data.squeeze(0).cpu().numpy()

        # Score
        rank = 1e20
        # find the highest ranking
        for i in range(5*index, 5*index + 5, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1

    return r1+r5+r10, (r1, r5, r10, medr)

def text_to_image(sentence_embs, img_embs):    
    scores = torch.mm(sentence_embs, img_embs.t())
    gt_score = scores.diag()
    count = scores.gt(gt_score.view(-1,1).expand_as(scores))
    ranks = count.sum(1) + 1 
    ranks = ranks.cpu().numpy()
    
    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 2)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 6)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 11)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks))
    meanr = ranks.mean() 
    
    return  (r1+r5+r10+medr), (r1, r5, r10, medr)