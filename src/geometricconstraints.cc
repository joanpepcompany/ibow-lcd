#include "ibow-lcd/geometricconstraints.h"

namespace ibow_lcd
{

GeomConstr::GeomConstr(const cv::Mat &q_img,
                       const cv::Mat &t_img,
                       const std::vector<cv::KeyPoint> &q_kps,
                       const std::vector<cv::KeyPoint> &t_kps,
                       const cv::Mat &q_descs_pts,
                       const cv::Mat &t_descs_pts,
                       const std::vector<cv::DMatch> &matches_pts,
                       const std::vector<KeyLine> &q_kls,
                       const std::vector<KeyLine> &t_kls,
                       const cv::Mat &q_descs_l,
                       const cv::Mat &t_descs_l,
                       const std::vector<cv::DMatch> &matches_l)
    : q_img_(q_img), t_img_(t_img), img_size_(q_img.size())

{
    bool debug_results_ = false;
    if (debug_results_)
    {
        cv::Mat init_line_matches = DrawMatches(t_kls, q_kls, t_img, q_img, matches_l);
        cv::Mat pts_matching;
        cv::drawMatches(q_img, q_kps, t_img, t_kps, matches_pts, pts_matching);
        cv::imshow("Pts image", pts_matching);

        cv::imshow("Initial line matches", init_line_matches);
        // Filter line matches using a global orientation btwn frames
    }
    std::vector<cv::DMatch> filt_matches, non_filt_matches;
    // FilterOrientMatches(q_kls, t_kls, matches_l, filt_matches, non_filt_matches);
    filt_matches = matches_l;

    // Compute inliers using intersection Intersection Points, Kpts and Line Endpoints
    //1. IntersectionPts of two lines
    std::vector<IntersectPt> v_p_intesect_pt;
    std::vector<IntersectPt> v_c_intesect_pt;

    std::vector<int> v_matches(q_kls.size(), NULL);
    for (size_t j = 0; j < filt_matches.size(); j++)
    {
        v_matches.at(filt_matches[j].queryIdx) = filt_matches[j].trainIdx;
    }

    detAndMatchIntersectPts(q_kls, t_kls,
                            v_matches,
                            v_p_intesect_pt, v_c_intesect_pt);

    // Convert v_keypoints to v_point2f to be used in fundamental Mat
    std::vector<cv::Point2f> v_p_intersect_kpts(v_p_intesect_pt.size());
    for (size_t j = 0; j < v_p_intesect_pt.size(); j++)
    {
        v_p_intersect_kpts[j] = v_p_intesect_pt[j].m_pt;
    }
    std::vector<cv::Point2f> v_c_intersect_kpts(v_c_intesect_pt.size());
    for (size_t j = 0; j < v_c_intesect_pt.size(); j++)
    {
        v_c_intersect_kpts[j] = v_c_intesect_pt[j].m_pt;
    }

    //2. Kpts
    // Match Keypoints between frames

    // Convert v_KeyPoint to v_Point2f to be used in fundamental Mat
    std::vector<cv::Point2f> v_c_kpts;
    std::vector<cv::Point2f> v_p_kpts;
    convertPoints(q_kps, t_kps, matches_pts, &v_c_kpts, &v_p_kpts);

    //3. Line EndPoints
    // Get Lines that have not been used before
    std::vector<int> no_intersect_matches = v_matches;
    // for (sizdebug_results_t j = 0; j < v_p_intesect_pt.size(); j++)
    // {
    // 	no_intersect_matches[v_p_intesect_pt[j].m_idx1] = NULL;
    // 	no_intersect_matches[v_p_intesect_pt[j].m_idx2] = NULL;
    // }

    std::vector<cv::Point2f> v_c_non_intersect_lines;
    std::vector<cv::Point2f> v_p_non_intersect_lines;
    // Convert Line endopoints to CV::Point2F to include them on findFundamentalMat.
    // Due to the line orientation can be different between frames the endpoints each line endpoints has been inserted with two diferent orientation (s-s/e-e) and (s-e/e-s)

    std::vector<cv::DMatch> v_cand_endpts_matches;
    for (size_t j = 0; j < no_intersect_matches.size(); j++)
    {
        if (no_intersect_matches[j])
        {
            cv::Point2f c_pt_1 = q_kls[j].getStartPoint();
            cv::Point2f c_pt_2 = q_kls[j].getEndPoint();

            cv::DMatch match;
            match.queryIdx = j;
            match.trainIdx = no_intersect_matches[j];
            v_cand_endpts_matches.push_back(match);

            v_c_non_intersect_lines.push_back(c_pt_1);
            v_c_non_intersect_lines.push_back(c_pt_2);
            v_c_non_intersect_lines.push_back(c_pt_2);
            v_c_non_intersect_lines.push_back(c_pt_1);

            cv::Point2f p_pt_1 = t_kls[no_intersect_matches[j]].getStartPoint();
            cv::Point2f p_pt_2 = t_kls[no_intersect_matches[j]].getEndPoint();

            v_p_non_intersect_lines.push_back(p_pt_1);
            v_p_non_intersect_lines.push_back(p_pt_2);
            v_p_non_intersect_lines.push_back(p_pt_2);
            v_p_non_intersect_lines.push_back(p_pt_1);
        }
    }

    // Combine 1 + 2 + 3 into a vector for FM computation
    // Combine IntersectPt Matching (1) and Ktps matching (2)
    std::vector<cv::Point2f> v_p_combined_pts = v_p_intersect_kpts;
    v_p_combined_pts.insert(v_p_combined_pts.end(), v_p_kpts.begin(), v_p_kpts.end());

    std::vector<cv::Point2f> v_c_combined_pts = v_c_intersect_kpts;
    v_c_combined_pts.insert(v_c_combined_pts.end(), v_c_kpts.begin(), v_c_kpts.end());

    // Add endpoints lines (3)
    int size_int_pts_and_kpts = v_c_combined_pts.size();

    v_p_combined_pts.insert(v_p_combined_pts.end(), v_p_non_intersect_lines.begin(), v_p_non_intersect_lines.end());

    v_c_combined_pts.insert(v_c_combined_pts.end(), v_c_non_intersect_lines.begin(), v_c_non_intersect_lines.end());

    // Compute FM with intersection Points and Keypoints
    double ep_dist = 1.0;
    double conf_prob = 0.99;
    std::vector<uchar> inliers_pts(v_c_combined_pts.size(), 0);

    cv::Mat FM = cv::findFundamentalMat(
        cv::Mat(v_c_combined_pts),
        cv::Mat(v_p_combined_pts), // Matching points
        CV_FM_RANSAC,              // RANSAC method
        ep_dist,                   // Distance to epipolar line
        conf_prob,                 // Confidence probability
        inliers_pts);              // Match status (inlier or outlier)

    //Split the inliers vector in IntersectionPts, Kpts and Endpoints
    //1. IntersectionPts of two lines
    std::vector<uchar> inliers_intersect_pts(inliers_pts.begin(), inliers_pts.begin() + v_p_intersect_kpts.size());

    int inters_inl_wout_dupl = 0;
    std::vector<int> line_match_aft_fm = v_matches;
    std::vector<int> line_match_after_fm(v_matches.size(), NULL);
    int intersect_inliers = 0;

    int idx_1 = 0;
    auto it1 = inliers_intersect_pts.begin();

    std::cerr << "v_p_intersect_kpts.size() : " << v_p_intersect_kpts.size() << std::endl;
    for (; it1 != inliers_intersect_pts.end(); it1++)
    {
        if (*it1)
        {
            if (!line_match_after_fm[v_c_intesect_pt[idx_1].m_idx1])
            {

                inters_inl_wout_dupl++;
                line_match_after_fm[v_c_intesect_pt[idx_1].m_idx1] = line_match_aft_fm[v_c_intesect_pt[idx_1].m_idx1];

                line_match_aft_fm[v_c_intesect_pt[idx_1].m_idx1] = NULL;
            }
            if (!line_match_after_fm[v_c_intesect_pt[idx_1].m_idx2])
            {

                inters_inl_wout_dupl++;
                line_match_after_fm[v_c_intesect_pt[idx_1].m_idx2] = line_match_aft_fm[v_c_intesect_pt[idx_1].m_idx2];
                line_match_aft_fm[v_c_intesect_pt[idx_1].m_idx2] = NULL;
            }
            intersect_inliers++;
        }
        idx_1++;
    }

    //2. Kpts
    std::vector<uchar> inliers_kpts_pts(inliers_pts.begin() + v_p_intersect_kpts.size(), inliers_pts.begin() + size_int_pts_and_kpts);

    auto it2 = inliers_kpts_pts.begin();
    int kpts_inliers = 0;

    for (; it2 != inliers_kpts_pts.end(); it2++)
    {
        if (*it2)
        {
            kpts_inliers++;
        }
    }

    //3. Endpoints of lines
    std::vector<uchar> inliers_endpts(inliers_pts.begin() + size_int_pts_and_kpts, inliers_pts.end());

    auto it3 = inliers_endpts.begin();
    bool inlier_endp_found = false;
    int endpts_inliers = 0;
    int line_endpoints_inliers = 0;
    int k = 0;
    int idx3 = 0;

    // Extract the surviving (inliers) matches and create a vector of Lines that are not matched
    std::vector<int> match_endpts_lines_int(v_matches.size(), NULL);

    for (; it3 != inliers_endpts.end(); it3++)
    {
        if (*it3)
        {
            inlier_endp_found = true;
            endpts_inliers++;
        }

        if ((idx3 + 1 - k * 4) > 3)
        {
            if (inlier_endp_found)
            {
                match_endpts_lines_int[v_cand_endpts_matches[k].queryIdx] = v_cand_endpts_matches[k].trainIdx;

                line_endpoints_inliers++;
                inlier_endp_found = false;
            }
            k++;
        }
        idx3++;
    }

    //Evaluate the number of Line inliers combining line endpoints and intersection points
    int inlier_total_line = 0;

    std::vector<int> line_match_after_fm_debug = line_match_after_fm;
    for (size_t j = 0; j < match_endpts_lines_int.size(); j++)
    {
        if (match_endpts_lines_int[j])
        {
            if (line_match_after_fm[j])
            {
                line_match_after_fm[j] = NULL;
            }
            inlier_total_line++;
        }
    }

    for (size_t j = 0; j < line_match_after_fm.size(); j++)
    {
        if (line_match_after_fm[j])
            inlier_total_line++;
    }

    // Evaluate if some lines non geometrical matched can be inliers using the proximity of the endpoints with the epipolar line. Line endpoints as well as the Fundamental Matrix previously computed are required.
    // std::vector<cv::DMatch> v_ctr_matches;
    // std::vector<cv::DMatch> v_ctr_non_matches;
    // std::vector<uchar> inliers_endpoints(line_match_aft_fm.size(), 0);

    // lFm->line_endpts_epip_geom_const(line_match_aft_fm, v_keylines.back(), v_keylines[v_keylines.size() - 2], FM, cam_matrix, inliers_endpoints, v_ctr_matches, v_ctr_non_matches);

    std::cerr << "*** Inliers resume ***" << std::endl;
    std::cerr << " - Amount of lines (Img1 / Img2) : " << t_kps.size() << " / " << q_kps.size() << std::endl;

    std::cerr << " - Ktps Inliers : " << kpts_inliers << " of " << v_p_kpts.size() << std::endl;

    std::cerr << " - Intersection Inliers Points : " << intersect_inliers << " of " << v_c_intersect_kpts.size() << std::endl;

    std::cerr << " - Intersection inliers lines without duplicade : " << inters_inl_wout_dupl << std::endl;

    std::cerr << "Endpoints inliers using FM : " << endpts_inliers << " of " << v_c_non_intersect_lines.size() << " total pts, which correspond to " << line_endpoints_inliers << " of " << v_c_non_intersect_lines.size() / 4 << " lines" << std::endl;

    // std::cerr << "Endpoints Matches non using FM : " << v_ctr_matches.size() << std::endl;

    std::cerr << " - Total inliers : " << kpts_inliers + inters_inl_wout_dupl + line_endpoints_inliers << " of " << matches_pts.size() + matches_l.size() << std::endl;

    std::cerr << "Total of lines inliers : " << inlier_total_line << std::endl;

    lines_inliers_ = inlier_total_line;
    pts_inliers_ = kpts_inliers;

    //Draw matches of endpoints
    if (debug_results_)
    {
        // cv::imshow("Current Image Line Detection", v_images_with_lines.back());
        // cv::imshow("Previous Image Line Detection", v_images_with_lines[v_images_with_lines.size() - 2]);
        // std::unique_ptr<LineManager> lineMangr(std::unique_ptr<LineManager>(new LineManager()));

        // cv::Mat matches_img = lineMangr->DrawMatches(
        // v_keylines[v_keylines.size() - 2],
        // v_keylines.back(),
        // v_imgs[v_imgs.size() - 2],
        // v_imgs.back(),
        // v_ctr_matches);

        // cv::Mat non_matches_img = lineMangr->DrawMatches(
        //     v_keylines[v_keylines.size() - 2],
        //     v_keylines.back(),
        //     v_imgs[v_imgs.size() - 2],
        //     v_imgs.back(), v_ctr_non_matches);

        // cv::imshow("Matched lines using enpoints with FM known previously", matches_img);
        // cv::imshow("NON-Matched lines using enpoints with FM known previously", non_matches_img);

        cv::Mat match_img_endpots_non_FM_int = DrawMatches(
            t_kls,
            q_kls,
            t_img,
            q_img, v_cand_endpts_matches);

        imshow("Lines Matches using line endpoints without knowledge of Fundamental Mat", match_img_endpots_non_FM_int);

        cv::Mat matches_inters_img_int = DrawMatches(
            t_kls,
            q_kls,
            t_img_,
            q_img_,
            line_match_after_fm_debug);

        cv::imshow("Line Matches using intersection points", matches_inters_img_int);

        cv::waitKey();
    }
}

void GeomConstr::FilterOrientMatches(const std::vector<KeyLine> q_lines,
                                     const std::vector<KeyLine> &tr_lines,
                                     const std::vector<cv::DMatch> &matches,
                                     std::vector<cv::DMatch> &filt_matches,
                                     std::vector<cv::DMatch> &non_filt_matches)
{
    //Compute line orientation change between pair lines of the two frames
    double global_angle = GlobalRotationImagePair(q_lines, tr_lines);
    std::cerr << "global_angle : " << global_angle << std::endl
              << std::endl;

    for (size_t i = 0; i < matches.size(); i++)
    {
        double angleDif = fabs(q_lines[matches[i].queryIdx].angle + global_angle - tr_lines[matches[i].trainIdx].angle) * 57.29;

        if (angleDif > 720)
            angleDif -= 720;

        if (angleDif > 360)
            angleDif -= 360;

        if (!((angleDif > 20) && (angleDif < 160) || (angleDif > 200) && (angleDif < 340)))
        {
            filt_matches.push_back(matches[i]);
        }

        else
            non_filt_matches.push_back(matches[i]);
    }
}

double GeomConstr::GlobalRotationImagePair(const std::vector<KeyLine> q_lines, const std::vector<KeyLine> &tr_lines)
{
    double TwoPI = 2 * M_PI;
    double rotationAngle = TwoPI;

    const unsigned int resolution_scale = 20;
    //step 1: compute the angle histogram of lines in the left and right images
    const unsigned int dim = 360 / resolution_scale; //number of the bins of histogram
    unsigned int index;                              //index in the histogram
    double direction;
    double scalar = 180 / (resolution_scale * 3.1415927); //used when compute the index
    double angleShift = (resolution_scale * M_PI) / 360;  //make sure zero is the middle of the interval

    std::array<double, dim> angleHistLeft;
    std::array<double, dim> angleHistRight;
    std::array<double, dim> lengthLeft; //lengthLeft[i] store the total line length of all the lines in the ith angle bin.
    std::array<double, dim> lengthRight;
    angleHistLeft.fill(0);
    angleHistRight.fill(0);
    lengthLeft.fill(0);
    lengthRight.fill(0);

    for (unsigned int linenum = 0; linenum < q_lines.size(); linenum++)
    {
        direction = q_lines[linenum].angle + M_PI + angleShift;
        direction = direction < TwoPI ? direction : (direction - TwoPI);
        index = floor(direction * scalar);
        angleHistLeft[index]++;
        lengthLeft[index] += q_lines[linenum].lineLength;
    }
    for (unsigned int linenum = 0; linenum < tr_lines.size(); linenum++)
    {
        direction = tr_lines[linenum].angle + M_PI + angleShift;
        direction = direction < TwoPI ? direction : (direction - TwoPI);
        index = floor(direction * scalar);
        angleHistRight[index]++;
        lengthRight[index] += tr_lines[linenum].lineLength;
    }
    arrayMultiRatio(angleHistLeft.data(), angleHistLeft.size(), (1 / getNormL2(angleHistLeft.data(), angleHistLeft.size())));
    arrayMultiRatio(angleHistRight.data(), angleHistRight.size(), (1 / getNormL2(angleHistRight.data(), angleHistRight.size())));
    arrayMultiRatio(lengthLeft.data(), lengthLeft.size(), (1 / getNormL2(lengthLeft.data(), lengthLeft.size())));
    arrayMultiRatio(lengthRight.data(), lengthRight.size(), (1 / getNormL2(lengthRight.data(), lengthRight.size())));

    //step 2: find shift to decide the approximate global rotation
    std::array<double, dim> difVec; //the difference vector between left histogram and shifted right histogram
    double minDif = 10;             //the minimal angle histogram difference
    double secondMinDif = 10;       //the second minimal histogram difference
    unsigned int minShift;          //the shift of right angle histogram when minimal difference achieved
    unsigned int secondMinShift;    //the shift of right angle histogram when second minimal difference achieved

    std::array<double, dim> lengthDifVec; //the length difference vector between left and right
    double minLenDif = 10;                //the minimal length difference
    double secondMinLenDif = 10;          //100		  //the second minimal length difference
    unsigned int minLenShift;             //the shift of right length vector when minimal length difference achieved
    unsigned int secondMinLenShift;       //the shift of right length vector when the second minimal length difference achieved

    double normOfVec;
    for (unsigned int shift = 0; shift < dim; shift++)
    {
        for (unsigned int j = 0; j < dim; j++)
        {
            index = j + shift;
            index = index < dim ? index : (index - dim);
            difVec[j] = angleHistLeft[j] - angleHistRight[index];
            lengthDifVec[j] = lengthLeft[j] - lengthRight[index];
        }
        //find the minShift and secondMinShift for angle histogram
        normOfVec = getNormL2(difVec.data(), difVec.size());
        if (normOfVec < secondMinDif)
        {
            if (normOfVec < minDif)
            {
                secondMinDif = minDif;
                secondMinShift = minShift;
                minDif = normOfVec;
                minShift = shift;
            }
            else
            {
                secondMinDif = normOfVec;
                secondMinShift = shift;
            }
        }
        //find the minLenShift and secondMinLenShift of length vector
        normOfVec = getNormL2(lengthDifVec.data(), lengthDifVec.size());
        if (normOfVec < secondMinLenDif)
        {
            if (normOfVec < minLenDif)
            {
                secondMinLenDif = minLenDif;
                secondMinLenShift = minLenShift;
                minLenDif = normOfVec;
                minLenShift = shift;
            }
            else
            {
                secondMinLenDif = normOfVec;
                secondMinLenShift = shift;
            }
        }
    }

    //first check whether there exist an approximate global rotation angle between image pair
    float AcceptableAngleHistogramDifference = 0.49;
    float AcceptableLengthVectorDifference = 0.4;
    if (minDif < AcceptableAngleHistogramDifference && minLenDif < AcceptableLengthVectorDifference)
    {
        rotationAngle = minShift * resolution_scale;
        if (rotationAngle > 90 && 360 - rotationAngle > 90)
        {
            //In most case we believe the rotation angle between two image pairs should belong to [-Pi/2, Pi/2]
            rotationAngle = rotationAngle - 180;
        }
        rotationAngle = rotationAngle * M_PI / 180;
    }
    return rotationAngle;
}

double GeomConstr::getNormL2(double *arr, int size)
{
    double result = 0;
    for (int i = 0; i < size; i++)
    {
        result = result + arr[i] * arr[i];
    }
    return sqrt(result);
}
void GeomConstr::arrayMultiRatio(double *arr, int size, double ratio)
{
    for (int i = 0; i < size; i++)
    {
        arr[i] = arr[i] * ratio;
    }
}

void GeomConstr::convertPoints(const std::vector<cv::KeyPoint> &query_kps,
                               const std::vector<cv::KeyPoint> &train_kps,
                               const std::vector<cv::DMatch> &matches,
                               std::vector<cv::Point2f> *query,
                               std::vector<cv::Point2f> *train)
{
    query->clear();
    train->clear();
    for (auto it = matches.begin(); it != matches.end(); it++)
    {
        // Get the position of query keypoints
        float x = query_kps[it->queryIdx].pt.x;
        float y = query_kps[it->queryIdx].pt.y;
        query->push_back(cv::Point2f(x, y));

        // Get the position of train keypoints
        x = train_kps[it->trainIdx].pt.x;
        y = train_kps[it->trainIdx].pt.y;
        train->push_back(cv::Point2f(x, y));
    }
}

void GeomConstr::detAndMatchIntersectPts(
    const std::vector<KeyLine> &q_kls,
    const std::vector<KeyLine> &t_kls,
    const std::vector<int> &v_matches,
    std::vector<IntersectPt> &v_p_inters_match_pt,
    std::vector<IntersectPt> &v_c_inters_match_pt)
{
    //Current Frame Detection
    std::vector<std::pair<int, int>> v_c_pair_cand;
    std::vector<cv::Point2f> v_c_intersect_pts;

    // Extract intersection points between two lines that:
    // a. Contains closer Endpoints distance of two lines
    // b. Contains a intersection pt close to one of the endpoints of the two lines
    float dist_th = 50.0;
    getLineIntersectCand(q_kls,
                         v_c_pair_cand, dist_th,
                         v_c_intersect_pts);

    //Previous Frame Detection

    std::vector<std::pair<int, int>> v_p_pair_cand;
    std::vector<cv::Point2f> v_p_intersect_pts;
    getLineIntersectCand(t_kls,
                         v_p_pair_cand, dist_th,
                         v_p_intersect_pts);

    // Find intersection points that have the two line matches in common between frames
    GetPtsWithTwoLineMatches(v_c_pair_cand, v_p_pair_cand, v_matches,
                             q_kls, t_kls,
                             v_c_intersect_pts, v_p_intersect_pts,
                             v_c_inters_match_pt, v_p_inters_match_pt);

    if (debug_results_)
    {
        // Debug results of Line and intersection point candidates
        // Line candidates
        cv::Mat gray_img;
        cv::cvtColor(q_img_, gray_img, CV_BGR2GRAY);
        // Filter candidates
        std::vector<KeyLine> v_cand_lines;
        for (size_t i = 0; i < v_c_inters_match_pt.size(); i++)
        {
            KeyLine line_1_cand = v_c_inters_match_pt[i].m_l1;
            v_cand_lines.push_back(line_1_cand);
            KeyLine line_2_cand = v_c_inters_match_pt[i].m_l2;
            v_cand_lines.push_back(line_2_cand);
        }

        cv::Mat line_img = draw2DLines(gray_img, v_cand_lines, "Debug line Candidates");

        // Intersection Points candidates
        std::vector<cv::KeyPoint> v_c_draw_kpts;
        for (size_t i = 0; i < v_c_inters_match_pt.size(); i++)
        {
            cv::KeyPoint single_kp;
            single_kp.pt = v_c_inters_match_pt[i].m_pt;
            v_c_draw_kpts.push_back(single_kp);
        }
        cv::Mat c_kpts_img;
        cv::drawKeypoints(line_img, v_c_draw_kpts, c_kpts_img);

        // Debug Previous Frame
        // Line candidates
        cv::Mat p_gray_img;
        cv::cvtColor(t_img_, p_gray_img, CV_BGR2GRAY);
        // Filter candidates
        std::vector<KeyLine> v_p_cand_lines;
        for (size_t i = 0; i < v_p_inters_match_pt.size(); i++)
        {
            KeyLine line_1_cand = v_p_inters_match_pt[i].m_l1;
            v_p_cand_lines.push_back(line_1_cand);
            KeyLine line_2_cand = v_p_inters_match_pt[i].m_l1;
            v_p_cand_lines.push_back(line_2_cand);
        }

        cv::Mat p_line_img = draw2DLines(p_gray_img, v_p_cand_lines, "Debug line Candidates");

        // Intersection Points candidates
        std::vector<cv::KeyPoint> v_p_draw_kpts;
        for (size_t i = 0; i < v_p_inters_match_pt.size(); i++)
        {
            cv::KeyPoint single_kp;
            single_kp.pt = v_p_inters_match_pt[i].m_pt;
            v_p_draw_kpts.push_back(single_kp);
        }

        cv::Mat p_kpts_img;
        cv::drawKeypoints(p_line_img, v_p_draw_kpts, p_kpts_img);

        cv::imshow("Debug Current Intersection Points candidates", c_kpts_img);
        cv::imshow("Debug Previous Intersection Points candidates", p_kpts_img);
    }
}

void GeomConstr::getLineIntersectCand(
    const std::vector<KeyLine> &v_keylines,
    std::vector<std::pair<int, int>> &pair_cand,
    const float &distance_th,
    std::vector<cv::Point2f> &v_intersect_pts)
{
    for (size_t i = 0; i < v_keylines.size(); i++)
    {
        for (size_t j = i + 1; j < v_keylines.size(); j++)
        {
            //Compute distance between EndPoints
            double dist = getEuclLineDistances(
                v_keylines[i].getStartPoint(), v_keylines[i].getEndPoint(),
                v_keylines[j].getStartPoint(), v_keylines[j].getEndPoint());
            // Compute Line intersection
            //Line 1
            float dx1 = v_keylines[i].startPointX - v_keylines[i].endPointX;
            float dy1 = v_keylines[i].startPointY - v_keylines[i].endPointY;

            if (dx1 == 0)
                dx1 = 0.001;

            if (dy1 == 0)
                dy1 = 0.001;

            float m1 = dy1 / dx1;
            // y = mx + c

            // intercept c = y - mx

            float c1 = v_keylines[i].endPointY - m1 * v_keylines[i].endPointX; // which is same as y2 - slope * x2

            //Line 2
            float dx2 = v_keylines[j].startPointX - v_keylines[j].endPointX;
            float dy2 = v_keylines[j].startPointY - v_keylines[j].endPointY;

            if (dx2 == 0)
                dx2 = 0.001;
            if (dy2 == 0)
                dy2 = 0.001;

            float m2 = dy2 / dx2;
            float c2 = v_keylines[j].endPointY - m2 * v_keylines[j].endPointX;

            double min_intersect_dist = 200.0, max_intersect_dist = 200.0;
            cv::Point2f intersect_pt;
            if ((m1 - m2) == 0)
                continue;

            else
            {
                float intersection_X = (c2 - c1) / (m1 - m2);
                float intersection_Y = m1 * intersection_X + c1;

                //Compute distance between two lines and a point
                intersect_pt = cv::Point2f(intersection_X, intersection_Y);

                cv::Mat m_intersect_dist = (Mat_<float>(1, 4) << abs(cv::norm(intersect_pt - v_keylines[i].getStartPoint())),
                                            abs(cv::norm(intersect_pt - v_keylines[i].getEndPoint())),
                                            abs(cv::norm(intersect_pt - v_keylines[j].getStartPoint())),
                                            abs(cv::norm(intersect_pt - v_keylines[j].getEndPoint())));

                // Get minimum value of the four distances
                cv::minMaxLoc(m_intersect_dist,
                              &min_intersect_dist, &max_intersect_dist);
            }

            if (dist < distance_th && min_intersect_dist < distance_th)
            {
                if (intersect_pt.x > 0 && intersect_pt.x < img_size_.width)
                    if (intersect_pt.y > 0 && intersect_pt.y < img_size_.height)
                    {
                        // bool debug = false;
                        // if (debug)
                        // {
                        //     // Debug results of Line and intersection point candidates
                        // // Line candidates
                        // std::vector<KeyLine> v_cand_lines;
                        // cv::Mat gray_img;
                        // cvtColor(img, gray_img, CV_BGR2GRAY);
                        // // Filter candidates

                        // KeyLine line_1_cand = v_keylines[i];
                        // v_cand_lines.push_back(line_1_cand);
                        // KeyLine line_2_cand = v_keylines[j];
                        // v_cand_lines.push_back(line_2_cand);

                        // cv::Mat line_img = draw2DLines(gray_img, v_cand_lines, "Debug line Candidates");

                        // // Intersection Points candidates
                        // std::vector<cv::KeyPoint> v_draw_kpts;
                        // cv::KeyPoint single_kp;
                        // single_kp.pt = intersect_pt;
                        // v_draw_kpts.push_back(single_kp);
                        // cv::Mat kpts_img;
                        // drawKeypoints(line_img, v_draw_kpts, kpts_img);

                        // cv::imshow("Debug candidates", kpts_img);
                        // cv::waitKey();
                        // }

                        pair_cand.push_back(std::pair<int, int>(i, j));
                        v_intersect_pts.push_back(intersect_pt);
                    }
            }
        }
    }
}

float GeomConstr::getEuclLineDistances(cv::Point l1_st, cv::Point l1_end,
                                       cv::Point l2_st, cv::Point l2_end)
{
    cv::Mat m_distances = (Mat_<float>(1, 4) << abs(cv::norm(l1_st - l2_st)), abs(cv::norm(l1_end - l2_end)), abs(cv::norm(l1_st - l2_end)), abs((cv::norm(l1_end - l2_st))));

    // Get minimum value of the four distances
    double min, max;
    cv::minMaxLoc(m_distances, &min, &max);
    return min;
}

void GeomConstr::GetPtsWithTwoLineMatches(const std::vector<std::pair<int, int>> &v_c_pair_cand,
                                          const std::vector<std::pair<int, int>> &v_p_pair_cand, const std::vector<int> &matches,
                                          const std::vector<KeyLine> &c_keylines,
                                          const std::vector<KeyLine> &p_keylines,
                                          const std::vector<cv::Point2f> &v_c_intersect_pts,
                                          const std::vector<cv::Point2f> &v_p_intersect_pts,
                                          std::vector<IntersectPt> &v_c_inters_match_pt,
                                          std::vector<IntersectPt> &v_p_inters_match_pt)
{
    // Find intersection point that has the two lines matched between frames
    for (size_t i = 0; i < v_c_pair_cand.size(); i++)
    {
        // If exist a match of this line between frames
        if (matches[v_c_pair_cand[i].first])
        {
            int idx = matches[v_c_pair_cand[i].first];
            // Look if in the second frame exist the match
            for (size_t k = 0; k < v_p_pair_cand.size(); k++)
            {
                // This match can be on the first or second pair column
                // Evluate first pair column
                if (v_p_pair_cand[k].first == idx)
                {
                    if (matches[v_c_pair_cand[i].second])
                    {
                        int idx_sec = matches[v_c_pair_cand[i].second];
                        if (idx_sec == v_p_pair_cand[k].second)
                        {
                            KeyLine l1_c = c_keylines[v_c_pair_cand[i].first];
                            KeyLine l2_c = c_keylines[v_c_pair_cand[i].second];
                            cv::Point2f c_intersect_pt = v_c_intersect_pts[i];
                            int c_idx1 = v_c_pair_cand[i].first;
                            int c_idx2 = v_c_pair_cand[i].second;

                            v_c_inters_match_pt.push_back(IntersectPt(c_intersect_pt, l1_c, l2_c, c_idx1, c_idx2));

                            KeyLine l1_p = p_keylines[v_p_pair_cand[k].first];
                            KeyLine l2_p = p_keylines[v_p_pair_cand[k].second];
                            cv::Point2f p_intersect_pt = v_p_intersect_pts[k];

                            int p_idx1 = v_p_pair_cand[k].first;
                            int p_idx2 = v_p_pair_cand[k].second;

                            v_p_inters_match_pt.push_back(IntersectPt(p_intersect_pt, l1_p, l2_p, p_idx1, p_idx2));

                            break;
                        }
                    }
                }

                //Evaluate second pair column
                else if (v_p_pair_cand[k].second == idx)
                {
                    if (matches[v_c_pair_cand[i].first])
                    {
                        int idx_sec = matches[v_c_pair_cand[i].first];
                        if (idx_sec == v_p_pair_cand[k].first)
                        {

                            KeyLine l1_c = c_keylines[v_c_pair_cand[i].first];
                            KeyLine l2_c = c_keylines[v_c_pair_cand[i].second];
                            cv::Point2f c_intersect_pt = v_c_intersect_pts[i];

                            int c_idx1 = v_c_pair_cand[i].first;
                            int c_idx2 = v_c_pair_cand[i].second;

                            v_c_inters_match_pt.push_back(IntersectPt(c_intersect_pt, l1_c, l2_c, c_idx1, c_idx2));

                            KeyLine l1_p = p_keylines[v_p_pair_cand[k].first];
                            KeyLine l2_p = p_keylines[v_p_pair_cand[k].second];
                            cv::Point2f p_intersect_pt = v_p_intersect_pts[k];

                            int p_idx1 = v_p_pair_cand[k].first;
                            int p_idx2 = v_p_pair_cand[k].second;

                            v_p_inters_match_pt.push_back(IntersectPt(p_intersect_pt, l1_p, l2_p, p_idx1, p_idx2));
                        }
                    }
                }
            }
        }
    }
}

cv::Mat GeomConstr::draw2DLines(const cv::Mat gray_img,
                                const std::vector<KeyLine> &keylines,
                                std::string img_name)
{
    cv::Mat line_img;
    if (gray_img.type() == CV_8UC1)
    {
        cvtColor(gray_img, line_img, CV_GRAY2RGB);
    }
    else
        line_img = gray_img.clone();

    for (size_t i = 0; i < keylines.size(); i++)
    {
        /* get a random color */
        int R = (rand() % (int)(255 + 1));
        int G = (rand() % (int)(255 + 1));
        int B = (rand() % (int)(255 + 1));

        // /* get ends of a line */
        cv::Point pt1 = cv::Point(int(keylines[i].startPointX), int(keylines[i].startPointY));
        cv::Point pt2 = cv::Point(int(keylines[i].endPointX), int(keylines[i].endPointY));

        /* draw line */
        line(line_img, pt1, pt2, cv::Scalar(B, G, R), 2);
    }

    return line_img;
}

cv::Mat GeomConstr::DrawMatches(const std::vector<KeyLine> &tr_lines,
                                const std::vector<KeyLine> &q_lines,
                                const cv::Mat &train_img,
                                const cv::Mat &query_img,
                                const std::vector<DMatch> &matchResult)
{
    cv::Mat tr_img = train_img.clone();
    cv::Mat q_img = query_img.clone();

    unsigned int imageWidth = tr_img.cols;
    unsigned int imageHeight = tr_img.rows;

    double ww1, ww2;
    int lineIDLeft;
    int lineIDRight;
    int lowest1 = 0, highest1 = 255;
    int range1 = (highest1 - lowest1) + 1;
    std::vector<unsigned int> r1(matchResult.size()), g1(matchResult.size()), b1(matchResult.size()); //the color of lines

    for (unsigned int pair = 0; pair < matchResult.size(); pair++)
    {
        r1[pair] = lowest1 + int(rand() % range1);
        g1[pair] = lowest1 + int(rand() % range1);
        b1[pair] = 255 - r1[pair];
        ww1 = 0.2 * (rand() % 5);
        ww2 = 1 - ww1;
        char buf[10];
        sprintf(buf, "%d ", pair);
        if (matchResult[pair].trainIdx < 0 || matchResult[pair].queryIdx < 0)
            continue;

        lineIDLeft = matchResult[pair].trainIdx;

        lineIDRight = matchResult[pair].queryIdx;
        cv::Point startPointL = cv::Point(int(tr_lines[lineIDLeft].startPointX), int(tr_lines[lineIDLeft].startPointY));
        cv::Point endPointL = cv::Point(int(tr_lines[lineIDLeft].endPointX), int(tr_lines[lineIDLeft].endPointY));
        cv::line(tr_img, startPointL, endPointL, CV_RGB(r1[pair], g1[pair], b1[pair]), 4, cv::LINE_AA, 0);

        putText(tr_img, std::to_string(lineIDLeft), cv::Point(startPointL.x, startPointL.y + 10), 1, 1, Scalar(255, 0, 0), 2, 0);

        cv::Point startPointR = cvPoint(int(q_lines[lineIDRight].startPointX), int(q_lines[lineIDRight].startPointY));

        cv::Point endPointR = cvPoint(int(q_lines[lineIDRight].endPointX), int(q_lines[lineIDRight].endPointY));

        cv::line(q_img, startPointR, endPointR, CV_RGB(r1[pair], g1[pair], b1[pair]), 4, cv::LINE_AA, 0);
        putText(q_img, std::to_string(lineIDRight), cv::Point(startPointR.x, startPointR.y + 10), 1, 1, Scalar(255, 0, 0), 2, 0);
    }

    cv::Mat cvResultColorImage1 = cv::Mat(cv::Size(imageWidth * 2, imageHeight), tr_img.type(), 3);
    cv::Mat cvResultColorImage2 = cv::Mat(cv::Size(imageWidth * 2, imageHeight), tr_img.type(), 3);
    cv::Mat cvResultColorImage = cv::Mat(cv::Size(imageWidth * 2, imageHeight), tr_img.type(), 3);
    cv::Mat roi = cvResultColorImage1(cv::Rect(0, 0, imageWidth, imageHeight));
    cv::resize(tr_img, roi, roi.size(), 0, 0, 0);

    cv::Mat roi2 = cvResultColorImage1(cv::Rect(imageWidth, 0, imageWidth, imageHeight));
    cv::resize(q_img, roi2, roi2.size(), 0, 0, 0);
    cvResultColorImage1.copyTo(cvResultColorImage2);

    for (unsigned int pair = 0; pair < matchResult.size(); pair++)
    {
        if (matchResult[pair].trainIdx < 0 || matchResult[pair].queryIdx < 0)
            continue;
        lineIDLeft = matchResult[pair].trainIdx;
        lineIDRight = matchResult[pair].queryIdx;
        cv::Point startPoint = cv::Point(int(tr_lines[lineIDLeft].startPointX), int(tr_lines[lineIDLeft].startPointY));
        cv::Point endPoint = cv::Point(int(q_lines[lineIDRight].startPointX + imageWidth), int(q_lines[lineIDRight].startPointY));
        cv::line(cvResultColorImage2, startPoint, endPoint, CV_RGB(r1[pair], g1[pair], b1[pair]), 1, cv::LINE_AA, 0);
    }
    cv::addWeighted(cvResultColorImage1, 0.5, cvResultColorImage2, 0.5, 0.0, cvResultColorImage, -1);

    float scale = 1.4;
    cv::resize(cvResultColorImage, cvResultColorImage, cv::Size(2 * imageWidth * scale, imageHeight * scale));

    return cvResultColorImage;
}

cv::Mat GeomConstr::DrawMatches(const std::vector<KeyLine> &tr_lines,
                                const std::vector<KeyLine> &q_lines,
                                const cv::Mat &train_img,
                                const cv::Mat &query_img,
                                const std::vector<int> &matchResult)
{
    cv::Mat tr_img = train_img.clone();
    cv::Mat q_img = query_img.clone();

    unsigned int imageWidth = tr_img.cols;
    unsigned int imageHeight = tr_img.rows;

    double ww1, ww2;
    int lineIDLeft;
    int lineIDRight;
    int lowest1 = 0, highest1 = 255;
    int range1 = (highest1 - lowest1) + 1;
    std::vector<unsigned int> r1(matchResult.size()), g1(matchResult.size()), b1(matchResult.size()); //the color of lines

    int total_matches = 0;
    int idx = 0;
    ;

    for (unsigned int pair = 0; pair < matchResult.size(); pair++)
    {
        if (!matchResult[pair])
            continue;
        r1[idx] = lowest1 + int(rand() % range1);
        g1[idx] = lowest1 + int(rand() % range1);
        b1[idx] = 255 - r1[idx];
        ww1 = 0.2 * (rand() % 5);
        ww2 = 1 - ww1;
        char buf[10];
        sprintf(buf, "%d ", pair);

        lineIDLeft = matchResult[pair];

        lineIDRight = pair;
        cv::Point startPointL = cv::Point(int(tr_lines[lineIDLeft].startPointX), int(tr_lines[lineIDLeft].startPointY));
        cv::Point endPointL = cv::Point(int(tr_lines[lineIDLeft].endPointX), int(tr_lines[lineIDLeft].endPointY));
        cv::line(tr_img, startPointL, endPointL, CV_RGB(r1[pair], g1[pair], b1[pair]), 4, cv::LINE_AA, 0);

        putText(tr_img, std::to_string(lineIDLeft), cv::Point(startPointL.x, startPointL.y + 10), 1, 1, Scalar(255, 0, 0), 2, 0);

        cv::Point startPointR = cvPoint(int(q_lines[lineIDRight].startPointX), int(q_lines[lineIDRight].startPointY));

        cv::Point endPointR = cvPoint(int(q_lines[lineIDRight].endPointX), int(q_lines[lineIDRight].endPointY));

        cv::line(q_img, startPointR, endPointR, CV_RGB(r1[pair], g1[pair], b1[pair]), 4, cv::LINE_AA, 0);
        putText(q_img, std::to_string(lineIDRight), cv::Point(startPointR.x, startPointR.y + 10), 1, 1, Scalar(255, 0, 0), 2, 0);
        idx++;
    }

    cv::Mat cvResultColorImage1 = cv::Mat(cv::Size(imageWidth * 2, imageHeight), tr_img.type(), 3);
    cv::Mat cvResultColorImage2 = cv::Mat(cv::Size(imageWidth * 2, imageHeight), tr_img.type(), 3);
    cv::Mat cvResultColorImage = cv::Mat(cv::Size(imageWidth * 2, imageHeight), tr_img.type(), 3);
    cv::Mat roi = cvResultColorImage1(cv::Rect(0, 0, imageWidth, imageHeight));
    cv::resize(tr_img, roi, roi.size(), 0, 0, 0);

    cv::Mat roi2 = cvResultColorImage1(cv::Rect(imageWidth, 0, imageWidth, imageHeight));
    cv::resize(q_img, roi2, roi2.size(), 0, 0, 0);
    cvResultColorImage1.copyTo(cvResultColorImage2);

    for (unsigned int pair = 0; pair < matchResult.size(); pair++)
    {
        if (!matchResult[pair])
            continue;
        lineIDLeft = matchResult[pair];
        lineIDRight = pair;
        cv::Point startPoint = cv::Point(int(tr_lines[lineIDLeft].startPointX), int(tr_lines[lineIDLeft].startPointY));
        cv::Point endPoint = cv::Point(int(q_lines[lineIDRight].startPointX + imageWidth), int(q_lines[lineIDRight].startPointY));
        cv::line(cvResultColorImage2, startPoint, endPoint, CV_RGB(r1[pair], g1[pair], b1[pair]), 1, cv::LINE_AA, 0);
    }
    cv::addWeighted(cvResultColorImage1, 0.5, cvResultColorImage2, 0.5, 0.0, cvResultColorImage, -1);

    float scale = 1.4;
    cv::resize(cvResultColorImage, cvResultColorImage, cv::Size(2 * imageWidth * scale, imageHeight * scale));

    return cvResultColorImage;
}

} // namespace ibow_lcd
