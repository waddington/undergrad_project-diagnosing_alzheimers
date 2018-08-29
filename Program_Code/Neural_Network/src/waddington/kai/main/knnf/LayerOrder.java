package waddington.kai.main.knnf;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * A small data structure that represents layer types in valid orders.
 * <p>
 * This is used to check that all layers are in a valid order when the network is checked.
 */
public class LayerOrder {

    /**
     * Represents the type of a layer.
     */
    private LayerType type;
    /**
     * A list containing all layer types that can directly precede this layer type.
     */
    private List<LayerType> antecedents;
    /**
     * A list containing all layer types that can come directly after this layer type.
     */
    private List<LayerType> consequents;

    /**
     * The only constructor for this class.
     * @param type The type of this layer.
     */
    LayerOrder(LayerType type) {
        this.type = type;

        this.antecedents = new ArrayList<>();
        this.consequents = new ArrayList<>();
    }

    /**
     * Adds valid antecedent(s) to this layer type.
     * @param types Vararg of layer types that will be added to this layer types' antecedents.
     */
    void addAntecedents(LayerType... types) {
        this.antecedents.addAll(Arrays.asList(types));
    }

    /**
     * Adds valid consequent(s) to this layer type.
     * @param types Vararg of layer types that will be added to this layer types' consequents.
     */
    void addConsequents(LayerType... types) {
        this.consequents.addAll(Arrays.asList(types));
    }

    /**
     * Checks if this layers antecedents contains a given layer type.
     * @param type A layer type to check for in this layers antecedents.
     * @return Returns true if this layer types antecedents contain the passed layer type.
     */
    boolean antecedentsContain(LayerType type) {
        if (this.antecedents.size() == 0)
            return false;

        boolean antecendentsContainType = false;

        for (LayerType t: this.antecedents)
            if (t == type)
                antecendentsContainType = true;

        return antecendentsContainType;
    }

    /**
     * Checks if this layers consequents contains a given layer type.
     * @param type A layer type to check for in this layers consequents.
     * @return Returns true if this layer types consequents contain the passed layer type.
     */
    public boolean consequentsContain(LayerType type) {
        if (this.consequents.size() == 0)
            return false;

        boolean consequentsContainType = false;

        for (LayerType t: this.consequents)
            if (t == type)
                consequentsContainType = true;

        return consequentsContainType;
    }
}